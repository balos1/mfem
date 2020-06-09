// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

// SUNDIALS vectors
#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_CUDA
#include <nvector/nvector_cuda.h>
#endif
#ifdef MFEM_USE_MPI
#include <nvector/nvector_mpiplusx.h>
#include <nvector/nvector_parallel.h>
#endif

// SUNDIALS linear solvers
#include <sunlinsol/sunlinsol_spgmr.h>

// Access SUNDIALS object's content pointer
#define GET_CONTENT(X) ( X->content )

using namespace std;

namespace mfem
{

static void* allocfn(size_t mem_size)
{
   Memory<double> *mem = new Memory<double>(mem_size/sizeof(double), Device::GetHostMemoryType());
   return *mem;
}

static void freefn(void* ptr)
{
   Memory<double> *mem =
      new Memory<double>((double*) ptr, 1, Device::GetHostMemoryType(), true);
   delete mem;
}

// ---------------------------------------------------------------------------
// SUNDIALS N_Vector interface functions
// ---------------------------------------------------------------------------

void SundialsNVector::_SetNvecDataAndSize_(long glob_size)
{
   // Set the N_Vector data and length from the Vector data and size.
   switch (GetNVectorID())
   {
      case SUNDIALS_NVEC_SERIAL:
      {
         MFEM_ASSERT(NV_OWN_DATA_S(x) == SUNFALSE, "invalid serial N_Vector");
         dbg("SUNDIALS_NVEC_SERIAL: h:%p d:%p", HostRead(), Read());
         if (Device::GetDeviceMemoryType() == mfem::MemoryType::DEVICE_DEBUG)
         {
            auto content = static_cast<N_VectorContent_Serial>(GET_CONTENT(x));
            content->ddata = HostReadWrite();
            content->data = ReadWrite();
         }
         else
         {
            NV_DATA_S(x) = HostReadWrite();
         }
         NV_LENGTH_S(x) = size;
         break;
      }
#ifdef MFEM_USE_CUDA
      case SUNDIALS_NVEC_CUDA:
      {
         auto content = static_cast<N_VectorContent_Cuda>(GET_CONTENT(x));
         // MFEM_ASSERT(content->own_data == SUNFALSE, "invalid cuda N_Vector");
         dbg("SUNDIALS_NVEC_CUDA: h:%p d:%p", HostRead(), Read());
         content->host_data = HostReadWrite();
         content->device_data = ReadWrite();
         content->length = size;
         break;
      }
#endif
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
      {
         MFEM_ASSERT(NV_OWN_DATA_P(x) == SUNFALSE, "invalid parallel N_Vector");
         NV_DATA_P(x) = HostReadWrite();
         NV_LOCLENGTH_P(x) = size;
         NV_GLOBLENGTH_P(x) = (glob_size == 0) ? size : glob_size;
         break;
      }
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(x)->local_vector;
         MFEM_ASSERT(hpv_local->owns_data == false, "invalid hypre N_Vector");
         hpv_local->data = HostReadWrite();
         hpv_local->size = size;
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << GetNVectorID() << " is not supported");
   }
}

void SundialsNVector::_SetDataAndSize_()
{
   // The SUNDIALS NVector owns the data if it created it.
   switch (GetNVectorID())
   {
      case SUNDIALS_NVEC_SERIAL:
      {
         double *h_ptr = NV_DDATA_S(x);
         double *d_ptr = NV_DATA_S(x);
         dbg("SUNDIALS_NVEC_SERIAL: h:%p, d:%p & size:%d", h_ptr, d_ptr, size);

         const bool known = mm.IsKnown(h_ptr);

         size = NV_LENGTH_S(x);
         if (Device::GetDeviceMemoryType() == mfem::MemoryType::DEVICE_DEBUG)
         {
            data.Wrap(h_ptr, d_ptr, size, Device::GetHostMemoryType(), false);
            if (known) { data.ClearOwnerFlags(); }
            UseDevice(true);
         }
         else
         {
            data.Wrap(NV_DATA_S(x), size, false);
            if (known) { data.ClearOwnerFlags(); }
         }
         break;
      }
#ifdef MFEM_USE_CUDA
      case SUNDIALS_NVEC_CUDA:
      {
         double *h_ptr = N_VGetHostArrayPointer_Cuda(x);
         double *d_ptr = N_VGetDeviceArrayPointer_Cuda(x);
         dbg("SUNDIALS_NVEC_CUDA: h:%p, d:%p & size:%d", h_ptr, d_ptr, size);

         const bool known = mm.IsKnown(h_ptr);

         size = N_VGetLength_Cuda(x);
         data.Wrap(h_ptr, d_ptr, size, Device::GetHostMemoryType(), false);
         if (known) { data.ClearOwnerFlags(); }
         UseDevice(true);
         break;
      }
#endif
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
      {
         const bool known = mm.IsKnown(NV_DATA_P(x));
         size = NV_LENGTH_S(x);
         data.Wrap(NV_DATA_P(x), NV_LOCLENGTH_P(x), false);
         if (known) { data.ClearOwnerFlags(); }
         break;
      }
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(x)->local_vector;
         const bool known = mm.IsKnown(NV_DATA_P(x));
         size = NV_LENGTH_S(x);
         data.Wrap(hpv_local->data, hpv_local->size, false);
         if (known) { data.ClearOwnerFlags(); }
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << GetNVectorID() << " is not supported");
   }
}

SundialsNVector::SundialsNVector()
   : Vector()
{
   // MFEM creates and owns the data,
   // and provides it to the SUNDIALS NVector.
   UseDevice(Device::IsAvailable());
   x = MakeNVector(UseDevice());
   own_NVector = 1;
}

SundialsNVector::SundialsNVector(N_Vector nv)
   : x(nv)
{
   _SetDataAndSize_();
   own_NVector = 0;
}

#ifdef MFEM_USE_MPI
   SundialsNVector::SundialsNVector(MPI_Comm comm)
      : Vector()
   {
      UseDevice(Device::IsAvailable());
      x = MakeNVector(comm, UseDevice());
   }

   SundialsNVector::SundialsNVector(MPI_Comm comm, int loc_size, long glob_size)
      : Vector(loc_size)
   {
      UseDevice(Device::IsAvailable());
      x = MakeNVector(comm, UseDevice(), data, loc_size, glob_size);
   }
#endif

SundialsNVector::~SundialsNVector()
{
   if (own_NVector)
   {
      N_VDestroy(x);
   }
}

void SundialsNVector::SetSize(int s, long glob_size)
{
   Vector::SetSize(s);
   _SetNvecDataAndSize_(glob_size);
}

void SundialsNVector::SetData(double *d)
{
   Vector::SetData(d);
   _SetNvecDataAndSize_();
}

void SundialsNVector::SetDataAndSize(double *d, int s, long glob_size)
{
   Vector::SetDataAndSize(d, s);
   _SetNvecDataAndSize_(glob_size);
}

N_Vector SundialsNVector::MakeNVector(bool use_device)
{
   N_Vector x;
#ifdef MFEM_USE_CUDA
   if (use_device && Device::GetDeviceMemoryType() != mfem::MemoryType::DEVICE_DEBUG)
   {
      x = N_VMake_Cuda(0, NULL, NULL);
   }
   else
   {
      // x = N_VNewEmpty_Serial(0);
      x = N_VMakeWithAllocator_Serial(0, NULL, allocfn, freefn);
   }
#else
   x = N_VNewEmpty_Serial(0);
#endif

   MFEM_VERIFY(x, "Error in SundialsNVector::MakeNVector.");

   return x;
}

#ifdef MFEM_USE_MPI

N_Vector SundialsNVector::MakeNVector(MPI_Comm comm, bool use_device)
{
   N_Vector x;

   if (comm == MPI_COMM_NULL)
   {
      x = MakeNVector(use_device);
   }
   else
   {
#ifdef MFEM_USE_CUDA
      if (use_device)
      {
         MFEM_ABORT("MPI+CUDA not yet supported by sundials interface");
      }
      else
      {
         x = N_VNewEmpty_Parallel(comm, 0, 0);
      }
#else
      x = N_VNewEmpty_Parallel(comm, 0, 0);
#endif // MFEM_USE_CUDA
   }

   MFEM_VERIFY(x, "Error in SundialsNVector::MakeNVector.");

   return x;
}

N_Vector SundialsNVector::MakeNVector(MPI_Comm comm, bool use_device, Memory<double> data,
                                      int loc_size, long glob_size)
{
   N_Vector x;

   if (comm == MPI_COMM_NULL)
   {
      x = MakeNVector(use_device);
   }
   else
   {
#ifdef MFEM_USE_CUDA
      if (use_device)
      {
         MFEM_ABORT("MPI+CUDA not yet supported by sundials interface");
      }
      else
      {
         x = N_VMake_Parallel(comm, loc_size, glob_size,
                              mfem::ReadWrite(data, loc_size, false));
      }
#else
      x = N_VMake_Parallel(comm, loc_size, glob_size,
                           mfem::ReadWrite(data, loc_size, false));
#endif // MFEM_USE_CUDA
   }

   MFEM_VERIFY(x, "Error in SundialsNVector::MakeNVector.");

   return x;
}
#endif // MFEM_USE_MPI


// ---------------------------------------------------------------------------
// SUNMatrix interface functions
// ---------------------------------------------------------------------------

// Return the matrix ID
static SUNMatrix_ID MatGetID(SUNMatrix)
{
   return (SUNMATRIX_CUSTOM);
}

static void MatDestroy(SUNMatrix A)
{
   if (A->content) { A->content = NULL; }
   if (A->ops) { free(A->ops); A->ops = NULL; }
   free(A); A = NULL;
   return;
}

// ---------------------------------------------------------------------------
// SUNLinearSolver interface functions
// ---------------------------------------------------------------------------

// Return the linear solver type
static SUNLinearSolver_Type LSGetType(SUNLinearSolver)
{
   return (SUNLINEARSOLVER_MATRIX_ITERATIVE);
}

static int LSFree(SUNLinearSolver LS)
{
   if (LS->content) { LS->content = NULL; }
   if (LS->ops) { free(LS->ops); LS->ops = NULL; }
   free(LS); LS = NULL;
   return (0);
}

// ---------------------------------------------------------------------------
// CVODE interface
// ---------------------------------------------------------------------------
int CVODESolver::RHS(realtype t, const N_Vector y, N_Vector ydot,
                     void *user_data)
{
   // At this point the up-to-date data for N_Vector y and ydot is on the device.
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_ydot(ydot);

   CVODESolver *self = static_cast<CVODESolver*>(user_data);

   // Compute y' = f(t, y)
   self->f->SetTime(t);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int CVODESolver::LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                             booleantype jok, booleantype *jcur, realtype gamma,
                             void*, N_Vector, N_Vector, N_Vector)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   const SundialsNVector mfem_fy(fy);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int CVODESolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                             N_Vector b, realtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   CVODESolver *self = static_cast<CVODESolver*>(GET_CONTENT(LS));
   // Solve the linear system
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

CVODESolver::CVODESolver(int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   Y = new SundialsNVector();
}

#ifdef MFEM_USE_MPI
CVODESolver::CVODESolver(MPI_Comm comm, int lmm)
   : lmm_type(lmm), step_mode(CV_NORMAL)
{
   Y = new SundialsNVector(comm);
}
#endif // MFEM_USE_MPI

void CVODESolver::Init(TimeDependentOperator &f_)
{
   // Initialize the base class
   ODESolver::Init(f_);

   // Get the vector length
   long local_size = f_.Height();
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    Y->Communicator());
#endif
   }

   // Get current time
   double t = f_.GetTime();

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last Init() call
      int resize = 0;
      if (!Parallel())
      {
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->Communicator());
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         CVodeFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create CVODE
      sundials_mem = CVodeCreate(lmm_type);
      MFEM_VERIFY(sundials_mem, "error in CVodeCreate()");

      // Initialize CVODE
      flag = CVodeInit(sundials_mem, CVODESolver::RHS, t, *Y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeInit()");

      // Attach the CVODESolver as user-defined data
      flag = CVodeSetUserData(sundials_mem, this);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetUserData()");

      // Set default tolerances
      flag = CVodeSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetSStolerances()");

      // Attach MFEM linear solver by default
      UseMFEMLinearSolver();
   }

   // Set the reinit flag to call CVodeReInit() in the next Step() call.
   reinit = true;
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   Y->MakeRef(x, 0, x.Size());
   MFEM_VERIFY(Y->Size() == x.Size(), "");

   // Reinitialize CVODE memory if needed
   if (reinit)
   {
      dbg("Reinit integrator");

      flag = CVodeReInit(sundials_mem, t, *Y);
      MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeReInit()");
      // reset flag
      reinit = false;
   }

   // Integrate the system
   dbg("Integrate the system");

   double tout = t + dt;
   flag = CVode(sundials_mem, tout, *Y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in CVode()");

   // Return the last incremental step size
   flag = CVodeGetLastStep(sundials_mem, &dt);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetLastStep()");
}

void CVODESolver::UseMFEMLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = CVODESolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = CVodeSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");

   // Set the linear system evaluation function
   flag = CVodeSetLinSysFn(sundials_mem, CVODESolver::LinSysSetup);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinSysFn()");
}

void CVODESolver::UseSundialsLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Create linear solver
   LSA = SUNLinSol_SPGMR(*Y, PREC_NONE, 0);
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = CVodeSetLinearSolver(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetLinearSolver()");
}

void CVODESolver::SetStepMode(int itask)
{
   step_mode = itask;
}

void CVODESolver::SetSStolerances(double reltol, double abstol)
{
   flag = CVodeSStolerances(sundials_mem, reltol, abstol);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSStolerances()");
}

void CVODESolver::SetMaxStep(double dt_max)
{
   flag = CVodeSetMaxStep(sundials_mem, dt_max);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxStep()");
}

void CVODESolver::SetMaxOrder(int max_order)
{
   flag = CVodeSetMaxOrd(sundials_mem, max_order);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeSetMaxOrd()");
}

void CVODESolver::PrintInfo() const
{
   long int nsteps, nfevals, nlinsetups, netfails;
   int      qlast, qcur;
   double   hinused, hlast, hcur, tcur;
   long int nniters, nncfails;

   // Get integrator stats
   flag = CVodeGetIntegratorStats(sundials_mem,
                                  &nsteps,
                                  &nfevals,
                                  &nlinsetups,
                                  &netfails,
                                  &qlast,
                                  &qcur,
                                  &hinused,
                                  &hlast,
                                  &hcur,
                                  &tcur);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetIntegratorStats()");

   // Get nonlinear solver stats
   flag = CVodeGetNonlinSolvStats(sundials_mem,
                                  &nniters,
                                  &nncfails);
   MFEM_VERIFY(flag == CV_SUCCESS, "error in CVodeGetNonlinSolvStats()");

   mfem::out <<
             "CVODE:\n"
             "num steps:            " << nsteps << "\n"
             "num rhs evals:        " << nfevals << "\n"
             "num lin setups:       " << nlinsetups << "\n"
             "num nonlin sol iters: " << nniters << "\n"
             "num nonlin conv fail: " << nncfails << "\n"
             "num error test fails: " << netfails << "\n"
             "last order:           " << qlast << "\n"
             "current order:        " << qcur << "\n"
             "initial dt:           " << hinused << "\n"
             "last dt:              " << hlast << "\n"
             "current dt:           " << hcur << "\n"
             "current t:            " << tcur << "\n" << endl;

   return;
}

CVODESolver::~CVODESolver()
{
   delete Y;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   SUNNonlinSolFree(NLS);
   CVodeFree(&sundials_mem);
}

// ---------------------------------------------------------------------------
// ARKStep interface
// ---------------------------------------------------------------------------

int ARKStepSolver::RHS1(realtype t, const N_Vector y, N_Vector ydot,
                        void *user_data)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_ydot(ydot);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute f(t, y) in y' = f(t, y) or fe(t, y) in y' = fe(t, y) + fi(t, y)
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   }
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int ARKStepSolver::RHS2(realtype t, const N_Vector y, N_Vector ydot,
                        void *user_data)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   SundialsNVector mfem_ydot(ydot);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(user_data);

   // Compute fi(t, y) in y' = fe(t, y) + fi(t, y)
   self->f->SetTime(t);
   self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

int ARKStepSolver::LinSysSetup(realtype t, N_Vector y, N_Vector fy, SUNMatrix A,
                               SUNMatrix, booleantype jok, booleantype *jcur,
                               realtype gamma,
                               void*, N_Vector, N_Vector, N_Vector)
{
   // Get data from N_Vectors
   const SundialsNVector mfem_y(y);
   const SundialsNVector mfem_fy(fy);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(A));

   // Compute the linear system
   self->f->SetTime(t);
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSetup(mfem_y, mfem_fy, jok, jcur, gamma));
}

int ARKStepSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                               N_Vector b, realtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the linear system
   if (self->rk_type == IMEX)
   {
      self->f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   }
   return (self->f->SUNImplicitSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassSysSetup(realtype t, SUNMatrix M,
                                void*, N_Vector, N_Vector, N_Vector)
{
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix system
   self->f->SetTime(t);
   return (self->f->SUNMassSetup());
}

int ARKStepSolver::MassSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector x,
                                N_Vector b, realtype tol)
{
   SundialsNVector mfem_x(x);
   const SundialsNVector mfem_b(b);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(LS));

   // Solve the mass matrix system
   return (self->f->SUNMassSolve(mfem_b, mfem_x, tol));
}

int ARKStepSolver::MassMult1(SUNMatrix M, N_Vector x, N_Vector v)
{
   const SundialsNVector mfem_x(x);
   SundialsNVector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(GET_CONTENT(M));

   // Compute the mass matrix-vector product
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

int ARKStepSolver::MassMult2(N_Vector x, N_Vector v, realtype t,
                             void* mtimes_data)
{
   const SundialsNVector mfem_x(x);
   SundialsNVector mfem_v(v);
   ARKStepSolver *self = static_cast<ARKStepSolver*>(mtimes_data);

   // Compute the mass matrix-vector product
   self->f->SetTime(t);
   return (self->f->SUNMassMult(mfem_x, mfem_v));
}

ARKStepSolver::ARKStepSolver(Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   Y = new SundialsNVector();
}

#ifdef MFEM_USE_MPI
ARKStepSolver::ARKStepSolver(MPI_Comm comm, Type type)
   : rk_type(type), step_mode(ARK_NORMAL),
     use_implicit(type == IMPLICIT || type == IMEX)
{
   Y = new SundialsNVector(comm);
}
#endif

void ARKStepSolver::Init(TimeDependentOperator &f_)
{
   // Initialize the base class
   ODESolver::Init(f_);

   // Get the vector length
   long local_size = f_.Height();
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    Y->Communicator());
#endif
   }

   // Get current time
   double t = f_.GetTime();

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last Init() call
      int resize = 0;
      if (!Parallel())
      {
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->Communicator());
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         ARKStepFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create ARKStep memory
      if (rk_type == IMPLICIT)
      {
         sundials_mem = ARKStepCreate(NULL, ARKStepSolver::RHS1, t, *Y);
      }
      else if (rk_type == EXPLICIT)
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, NULL, t, *Y);
      }
      else
      {
         sundials_mem = ARKStepCreate(ARKStepSolver::RHS1, ARKStepSolver::RHS2,
                                      t, *Y);
      }
      MFEM_VERIFY(sundials_mem, "error in ARKStepCreate()");

      // Attach the ARKStepSolver as user-defined data
      flag = ARKStepSetUserData(sundials_mem, this);
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetUserData()");

      // Set default tolerances
      flag = ARKStepSStolerances(sundials_mem, default_rel_tol, default_abs_tol);
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetSStolerances()");

      // If implicit, attach MFEM linear solver by default
      if (use_implicit) { UseMFEMLinearSolver(); }
   }

   // Set the reinit flag to call ARKStepReInit() in the next Step() call.
   reinit = true;
}

void ARKStepSolver::Step(Vector &x, double &t, double &dt)
{
   Y->SetData(x.GetMemory());

   MFEM_VERIFY(Y->Size() == x.Size(), "");

   // Reinitialize ARKStep memory if needed
   if (reinit)
   {
      if (rk_type == IMPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, NULL, ARKStepSolver::RHS1, t, *Y);
      }
      else if (rk_type == EXPLICIT)
      {
         flag = ARKStepReInit(sundials_mem, ARKStepSolver::RHS1, NULL, t, *Y);
      }
      else
      {
         flag = ARKStepReInit(sundials_mem,
                              ARKStepSolver::RHS1, ARKStepSolver::RHS2, t, *Y);
      }
      MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepReInit()");

      // reset flag
      reinit = false;
   }

   // Integrate the system
   double tout = t + dt;
   flag = ARKStepEvolve(sundials_mem, tout, *Y, &t, step_mode);
   MFEM_VERIFY(flag >= 0, "error in ARKStepEvolve()");

   // Return the last incremental step size
   flag = ARKStepGetLastStep(sundials_mem, &dt);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetLastStep()");
}

void ARKStepSolver::UseMFEMLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = ARKStepSolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = ARKStepSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

   // Set the linear system evaluation function
   flag = ARKStepSetLinSysFn(sundials_mem, ARKStepSolver::LinSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinSysFn()");
}

void ARKStepSolver::UseSundialsLinearSolver()
{
   // Free any existing matrix and linear solver
   if (A != NULL)   { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Create linear solver
   LSA = SUNLinSol_SPGMR(*Y, PREC_NONE, 0);
   MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = ARKStepSetLinearSolver(sundials_mem, LSA, NULL);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");
}

void ARKStepSolver::UseMFEMMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(M); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Wrap linear solver as SUNLinearSolver and SUNMatrix
   LSM = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSM, "error in SUNLinSolNewEmpty()");

   LSM->content      = this;
   LSM->ops->gettype = LSGetType;
   LSM->ops->solve   = ARKStepSolver::MassSysSolve;
   LSA->ops->free    = LSFree;

   M = SUNMatNewEmpty();
   MFEM_VERIFY(M, "error in SUNMatNewEmpty()");

   M->content      = this;
   M->ops->getid   = SUNMatGetID;
   M->ops->matvec  = ARKStepSolver::MassMult1;
   M->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = ARKStepSetMassLinearSolver(sundials_mem, LSM, M, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetLinearSolver()");

   // Set the linear system function
   flag = ARKStepSetMassFn(sundials_mem, ARKStepSolver::MassSysSetup);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassFn()");
}

void ARKStepSolver::UseSundialsMassLinearSolver(int tdep)
{
   // Free any existing matrix and linear solver
   if (M != NULL)   { SUNMatDestroy(A); M = NULL; }
   if (LSM != NULL) { SUNLinSolFree(LSM); LSM = NULL; }

   // Create linear solver
   LSM = SUNLinSol_SPGMR(*Y, PREC_NONE, 0);
   MFEM_VERIFY(LSM, "error in SUNLinSol_SPGMR()");

   // Attach linear solver
   flag = ARKStepSetMassLinearSolver(sundials_mem, LSM, NULL, tdep);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassLinearSolver()");

   // Attach matrix multiplication function
   flag = ARKStepSetMassTimes(sundials_mem, NULL, ARKStepSolver::MassMult2,
                              this);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMassTimes()");
}

void ARKStepSolver::SetStepMode(int itask)
{
   step_mode = itask;
}

void ARKStepSolver::SetSStolerances(double reltol, double abstol)
{
   flag = ARKStepSStolerances(sundials_mem, reltol, abstol);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSStolerances()");
}

void ARKStepSolver::SetMaxStep(double dt_max)
{
   flag = ARKStepSetMaxStep(sundials_mem, dt_max);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetMaxStep()");
}

void ARKStepSolver::SetOrder(int order)
{
   flag = ARKStepSetOrder(sundials_mem, order);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetOrder()");
}

void ARKStepSolver::SetERKTableNum(int table_num)
{
   flag = ARKStepSetTableNum(sundials_mem, -1, table_num);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIRKTableNum(int table_num)
{
   flag = ARKStepSetTableNum(sundials_mem, table_num, -1);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetIMEXTableNum(int etable_num, int itable_num)
{
   flag = ARKStepSetTableNum(sundials_mem, itable_num, etable_num);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetTableNum()");
}

void ARKStepSolver::SetFixedStep(double dt)
{
   flag = ARKStepSetFixedStep(sundials_mem, dt);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepSetFixedStep()");
}

void ARKStepSolver::PrintInfo() const
{
   long int nsteps, expsteps, accsteps, step_attempts;
   long int nfe_evals, nfi_evals;
   long int nlinsetups, netfails;
   double   hinused, hlast, hcur, tcur;
   long int nniters, nncfails;

   // Get integrator stats
   flag = ARKStepGetTimestepperStats(sundials_mem,
                                     &expsteps,
                                     &accsteps,
                                     &step_attempts,
                                     &nfe_evals,
                                     &nfi_evals,
                                     &nlinsetups,
                                     &netfails);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetTimestepperStats()");

   flag = ARKStepGetStepStats(sundials_mem,
                              &nsteps,
                              &hinused,
                              &hlast,
                              &hcur,
                              &tcur);

   // Get nonlinear solver stats
   flag = ARKStepGetNonlinSolvStats(sundials_mem,
                                    &nniters,
                                    &nncfails);
   MFEM_VERIFY(flag == ARK_SUCCESS, "error in ARKStepGetNonlinSolvStats()");

   mfem::out <<
             "ARKStep:\n"
             "num steps:                 " << nsteps << "\n"
             "num exp rhs evals:         " << nfe_evals << "\n"
             "num imp rhs evals:         " << nfi_evals << "\n"
             "num lin setups:            " << nlinsetups << "\n"
             "num nonlin sol iters:      " << nniters << "\n"
             "num nonlin conv fail:      " << nncfails << "\n"
             "num steps attempted:       " << step_attempts << "\n"
             "num acc limited steps:     " << accsteps << "\n"
             "num exp limited stepfails: " << expsteps << "\n"
             "num error test fails:      " << netfails << "\n"
             "initial dt:                " << hinused << "\n"
             "last dt:                   " << hlast << "\n"
             "current dt:                " << hcur << "\n"
             "current t:                 " << tcur << "\n" << endl;

   return;
}

ARKStepSolver::~ARKStepSolver()
{
   delete Y;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   SUNNonlinSolFree(NLS);
   ARKStepFree(&sundials_mem);
}

// ---------------------------------------------------------------------------
// KINSOL interface
// ---------------------------------------------------------------------------

// Wrapper for evaluating the nonlinear residual F(u) = 0
int KINSolver::Mult(const N_Vector u, N_Vector fu, void *user_data)
{
   const SundialsNVector mfem_u(u);
   SundialsNVector mfem_fu(fu);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Compute the non-linear action F(u).
   self->oper->Mult(mfem_u, mfem_fu);

   // Return success
   return 0;
}

// Wrapper for computing Jacobian-vector products
int KINSolver::GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                            booleantype *new_u, void *user_data)
{
   const SundialsNVector mfem_v(v);
   SundialsNVector mfem_Jv(Jv);
   KINSolver *self = static_cast<KINSolver*>(user_data);

   // Update Jacobian information if needed
   if (*new_u)
   {
      const SundialsNVector mfem_u(u);
      self->jacobian = &self->oper->GetGradient(mfem_u);
      *new_u = SUNFALSE;
   }

   // Compute the Jacobian-vector product
   self->jacobian->Mult(mfem_v, mfem_Jv);

   // Return success
   return 0;
}

// Wrapper for evaluating linear systems J u = b
int KINSolver::LinSysSetup(N_Vector u, N_Vector, SUNMatrix J,
                           void *, N_Vector , N_Vector )
{
   const SundialsNVector mfem_u(u);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(J));

   // Update the Jacobian
   self->jacobian = &self->oper->GetGradient(mfem_u);

   // Set the Jacobian solve operator
   self->prec->SetOperator(*self->jacobian);

   // Return success
   return (0);
}

// Wrapper for solving linear systems J u = b
int KINSolver::LinSysSolve(SUNLinearSolver LS, SUNMatrix, N_Vector u,
                           N_Vector b, realtype)
{
   SundialsNVector mfem_u(u), mfem_b(b);
   KINSolver *self = static_cast<KINSolver*>(GET_CONTENT(LS));

   // Solve for u = [J(u)]^{-1} b, maybe approximately.
   self->prec->Mult(mfem_b, mfem_u);

   // Return success
   return (0);
}

KINSolver::KINSolver(int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL), maa(0)
{
   // Allocate empty serial N_Vectors
   Y = new SundialsNVector();
   y_scale = new SundialsNVector();
   f_scale = new SundialsNVector();

   // Default abs_tol and print_level
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
   print_level = 0;
}

#ifdef MFEM_USE_MPI
KINSolver::KINSolver(MPI_Comm comm, int strategy, bool oper_grad)
   : global_strategy(strategy), use_oper_grad(oper_grad), y_scale(NULL),
     f_scale(NULL), jacobian(NULL), maa(0)
{
   Y = new SundialsNVector(comm);
   y_scale = new SundialsNVector(comm);
   f_scale = new SundialsNVector(comm);

   // Default abs_tol and print_level
   abs_tol     = pow(UNIT_ROUNDOFF, 1.0/3.0);
   print_level = 0;
}
#endif


void KINSolver::SetOperator(const Operator &op)
{
   // Initialize the base class
   NewtonSolver::SetOperator(op);
   jacobian = NULL;

   // Get the vector length
   long local_size = height;
#ifdef MFEM_USE_MPI
   long global_size;
#endif

   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    Y->Communicator());
#endif
   }

   if (sundials_mem)
   {
      // Check if the problem size has changed since the last SetOperator call
      int resize = 0;
      if (!Parallel())
      {
         resize = (Y->Size() != local_size);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int l_resize = (Y->Size() != local_size) ||
                        (saved_global_size != global_size);
         MPI_Allreduce(&l_resize, &resize, 1, MPI_INT, MPI_LOR,
                       Y->Communicator());
#endif
      }

      // Free existing solver memory and re-create with new vector size
      if (resize)
      {
         KINFree(&sundials_mem);
         sundials_mem = NULL;
      }
   }

   if (!sundials_mem)
   {
      if (!Parallel())
      {
         Y->SetSize(local_size);
      }
#ifdef MFEM_USE_MPI
      else
      {
         Y->SetSize(local_size, global_size);
         y_scale->SetSize(local_size, global_size);
         f_scale->SetSize(local_size, global_size);
         saved_global_size = global_size;
      }
#endif

      // Create the solver memory
      sundials_mem = KINCreate();
      MFEM_VERIFY(sundials_mem, "Error in KINCreate().");

      // Set number of acceleration vectors
      if (maa > 0)
      {
         flag = KINSetMAA(sundials_mem, maa);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");
      }

      // Initialize KINSOL
      flag = KINInit(sundials_mem, KINSolver::Mult, *Y);
      MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINInit()");

      // Attach the KINSolver as user-defined data
      flag = KINSetUserData(sundials_mem, this);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetUserData()");

      // Set the linear solver
      if (prec)
      {
         KINSolver::SetSolver(*prec);
      }
      else
      {
         // Free any existing linear solver
         if (A != NULL) { SUNMatDestroy(A); A = NULL; }
         if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

         LSA = SUNLinSol_SPGMR(*Y, PREC_NONE, 0);
         MFEM_VERIFY(LSA, "error in SUNLinSol_SPGMR()");

         flag = KINSetLinearSolver(sundials_mem, LSA, NULL);
         MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetLinearSolver()");

         // Set Jacobian-vector product function
         if (use_oper_grad)
         {
            flag = KINSetJacTimesVecFn(sundials_mem, KINSolver::GradientMult);
            MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetJacTimesVecFn()");
         }
      }
   }
}

void KINSolver::SetSolver(Solver &solver)
{
   // Store the solver
   prec = &solver;

   // Free any existing linear solver
   if (A != NULL) { SUNMatDestroy(A); A = NULL; }
   if (LSA != NULL) { SUNLinSolFree(LSA); LSA = NULL; }

   // Wrap KINSolver as SUNLinearSolver and SUNMatrix
   LSA = SUNLinSolNewEmpty();
   MFEM_VERIFY(LSA, "error in SUNLinSolNewEmpty()");

   LSA->content      = this;
   LSA->ops->gettype = LSGetType;
   LSA->ops->solve   = KINSolver::LinSysSolve;
   LSA->ops->free    = LSFree;

   A = SUNMatNewEmpty();
   MFEM_VERIFY(A, "error in SUNMatNewEmpty()");

   A->content      = this;
   A->ops->getid   = MatGetID;
   A->ops->destroy = MatDestroy;

   // Attach the linear solver and matrix
   flag = KINSetLinearSolver(sundials_mem, LSA, A);
   MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetLinearSolver()");

   // Set the Jacobian evaluation function
   flag = KINSetJacFn(sundials_mem, KINSolver::LinSysSetup);
   MFEM_VERIFY(flag == KIN_SUCCESS, "error in KINSetJacFn()");
}

void KINSolver::SetScaledStepTol(double sstol)
{
   flag = KINSetScaledStepTol(sundials_mem, sstol);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetScaledStepTol()");
}

void KINSolver::SetMaxSetupCalls(int max_calls)
{
   flag = KINSetMaxSetupCalls(sundials_mem, max_calls);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMaxSetupCalls()");
}

void KINSolver::SetMAA(int m_aa)
{
   // Store internally as maa must be set before calling KINInit() to
   // set the maximum acceleration space size.
   maa = m_aa;
   if (sundials_mem)
   {
      flag = KINSetMAA(sundials_mem, maa);
      MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetMAA()");
   }
}

// Compute the scaling vectors and solve nonlinear system
void KINSolver::Mult(const Vector&, Vector &x) const
{
   // residual norm tolerance
   double tol;

   // Uses c = 1, corresponding to x_scale.
   c = 1.0;

   if (!iterative_mode) { x = 0.0; }

   // For relative tolerance, r = 1 / |residual(x)|, corresponding to fx_scale.
   if (rel_tol > 0.0)
   {

      oper->Mult(x, r);

      // Note that KINSOL uses infinity norms.
      double norm = r.Normlinf();
#ifdef MFEM_USE_MPI
      if (Parallel())
      {
         double lnorm = norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_MAX,
                       Y->Communicator());
      }
#endif
      if (abs_tol > rel_tol * norm)
      {
         r = 1.0;
         tol = abs_tol;
      }
      else
      {
         r =  1.0 / norm;
         tol = rel_tol;
      }
   }
   else
   {
      r = 1.0;
      tol = abs_tol;
   }

   // Set the residual norm tolerance
   flag = KINSetFuncNormTol(sundials_mem, tol);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINSetFuncNormTol()");

   // Solve the nonlinear system by calling the other Mult method
   KINSolver::Mult(x, c, r);
}

// Solve the nonlinear system using the provided scaling vectors
void KINSolver::Mult(Vector &x,
                     const Vector &x_scale, const Vector &fx_scale) const
{
   flag = KINSetNumMaxIters(sundials_mem, max_iter);
   MFEM_ASSERT(flag == KIN_SUCCESS, "KINSetNumMaxIters() failed!");

   Y->SetData(x.GetMemory());
   y_scale->SetData(const_cast<Memory<double>&>(x_scale.GetMemory()));
   f_scale->SetData(const_cast<Memory<double>&>(fx_scale.GetMemory()));

   if (!Parallel())
   {
      flag = KINSetPrintLevel(sundials_mem, print_level);
      MFEM_VERIFY(flag == KIN_SUCCESS, "KINSetPrintLevel() failed!");
   }
   else
   {

#ifdef MFEM_USE_MPI
      int rank;
      MPI_Comm_rank(Y->Communicator(), &rank);
      if (rank == 0)
      {
         flag = KINSetPrintLevel(sundials_mem, print_level);
         MFEM_VERIFY(flag == KIN_SUCCESS, "KINSetPrintLevel() failed!");
      }
#endif

   }

   if (!iterative_mode) { x = 0.0; }

   // Solve the nonlinear system
   flag = KINSol(sundials_mem, *Y, global_strategy, *y_scale, *f_scale);
   converged = (flag >= 0);

   // Get number of nonlinear iterations
   long int tmp_nni;
   flag = KINGetNumNonlinSolvIters(sundials_mem, &tmp_nni);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINGetNumNonlinSolvIters()");
   final_iter = (int) tmp_nni;

   // Get the residual norm
   flag = KINGetFuncNorm(sundials_mem, &final_norm);
   MFEM_ASSERT(flag == KIN_SUCCESS, "error in KINGetFuncNorm()");
}

KINSolver::~KINSolver()
{
   delete Y;
   delete y_scale;
   delete f_scale;
   SUNMatDestroy(A);
   SUNLinSolFree(LSA);
   KINFree(&sundials_mem);
}

} // namespace mfem

#endif // MFEM_USE_SUNDIALS
