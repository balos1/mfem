#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

#define ZERO_RHO_BC
#define ZERO_IFND_BC

//#define ELIMINATE_REDUNDANT_VARS
//#define EQUATE_REDUNDANT_VARS

#define MIXED_MATRIX_NO_TRANSPOSE


#define SPARSE_ASDCOMPLEX
#define HYPRE_PARALLEL_ASDCOMPLEX


#define PENALTY_U_S 0.0


void test1_E_exact(const Vector &x, Vector &E);
void test1_RHS_exact(const Vector &x, Vector &f);
void test1_f_exact_0(const Vector &x, Vector &f);
void test1_f_exact_1(const Vector &x, Vector &f);
void test2_E_exact(const Vector &x, Vector &E);
void test2_RHS_exact(const Vector &x, Vector &f);
void test2_f_exact_0(const Vector &x, Vector &f);
void test2_f_exact_1(const Vector &x, Vector &f);
double test2_rho_exact_0(const Vector &x);
double test2_rho_exact_1(const Vector &x);
SparseMatrix* GetSparseMatrixFromOperator(Operator *op);
void PrintNonzerosOfInterfaceOperator(Operator const& op, const int tsize0, const int tsize1, set<int> const& tdofsbdry0,
				      set<int> const& tdofsbdry1, const int nprocs, const int rank);

hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat);
SparseMatrix* GatherHypreParMatrix(HypreParMatrix *A);
SparseMatrix* ReceiveSparseMatrix(const int source);
void SendSparseMatrix(SparseMatrix *S, const int dest);
HypreParMatrix* AddSubdomainMatrixAndInterfaceMatrix(MPI_Comm ifcomm, HypreParMatrix *A, HypreParMatrix *I, std::vector<int> & inj,
						     std::vector<int> & ginj,
						     ParFiniteElementSpace *ifespace, ParFiniteElementSpace *ifespace2=NULL,
						     const bool adding=true, const double cI=0.0, const double cA=1.0, const bool injRows=true);

HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
					       Array2D<double> const& coefficient);

Operator* CreateStrumpackMatrixFromHypreBlocks(MPI_Comm comm, const Array<int> & offsets, const Array2D<HypreParMatrix*> & blocks,
					       const Array2D<std::vector<int>*> & leftInjection,
					       const Array2D<std::vector<int>*> & rightInjection,
					       const Array2D<double> & coefficient);

void FindBoundaryTrueDOFs(ParFiniteElementSpace *pfespace, set<int>& tdofsBdry);
bool FacesCoincideGeometrically(ParMesh *volumeMesh, const int face, ParMesh *surfaceMesh, const int elem);
void PrintMeshBoundingBox(ParMesh *mesh);
void SetInterfaceToSurfaceDOFMap(ParFiniteElementSpace *ifespace, ParFiniteElementSpace *fespace, ParFiniteElementSpace *fespaceGlobal,
				 ParMesh *pmesh, const int sdAttribute, const std::set<int>& pmeshFacesInInterface,
				 const std::set<int>& pmeshEdgesInInterface, const std::set<int>& pmeshVerticesInInterface, 
				 const FiniteElementCollection *fec, std::vector<int>& dofmap, //std::vector<int>& fdofmap,
				 std::vector<int>& gdofmap);

void SetDomainDofsFromSubdomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & ssd, Vector & s);
void SetSubdomainDofsFromDomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & s, Vector & ssd);

// TODO: combine SetInjectionOperator and InjectionOperator as one class?
class SetInjectionOperator : public Operator
{
private:
  std::set<int> *id;
  
public:
  SetInjectionOperator(const int height, std::set<int> *a) : Operator(height, a->size()), id(a)
  {
    MFEM_VERIFY(height >= width, "SetInjectionOperator constructor");
  }

  ~SetInjectionOperator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    y = 0.0;

    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[*it] = x[i];
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[i] = x[*it];
  }
};

class InjectionOperator : public Operator
{
private:
  int *id;  // Size should be fullWidth, mapping to true subdomain DOF's.
  //mutable ParGridFunction gf;
  //int fullWidth;
  int m_nprocs, m_rank;
  
  std::vector<int> m_numTrueSD;
  std::vector<int> m_numLocalTrueSDmappedFromProc;
  std::vector<int> m_alliftsize;  // TODO: not needed as a class member.
  std::vector<int> m_alltrueSD;

  std::vector<int> m_iftToSDrank;  // map from interface true DOF to rank owning the corresponding subdomain true DOF.
  
  mutable std::vector<double> m_recv, m_send;
  std::vector<int> m_scnt, m_sdspl, m_rcnt, m_rdspl;

  MPI_Comm m_comm;
public:

  InjectionOperator(ParFiniteElementSpace *subdomainSpace, ParFiniteElementSpace *interfaceSpace, int *a,
		    std::vector<int> const& gdofmap);
  
  ~InjectionOperator()
  {
  }

  void SetComm(MPI_Comm comm)
  {
    m_comm = comm;
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    //gf.SetFromTrueDofs(x);

    // Given an input vector x of local true interface DOF's, set the output vector y of local true SD DOF's, which may require values from other processes. 

    MFEM_VERIFY(m_alltrueSD.size() == m_recv.size(), "");
    MFEM_VERIFY(x.Size() == m_send.size(), "");

    std::vector<int> cnt;
    cnt.assign(m_nprocs, 0);

    //for (int ifp=0; ifp<m_nprocs; ++ifp)
    for (int i=0; i<x.Size(); ++i) // loop over local true interface DOF's
      {
	m_send[m_sdspl[m_iftToSDrank[i]] + cnt[m_iftToSDrank[i]]] = x[i];
	cnt[m_iftToSDrank[i]]++;
      }

    { // debugging
      for (int i=0; i<m_nprocs; ++i)
	{
	  MFEM_VERIFY(cnt[i] == m_scnt[i], "");
	}
    }
    
    MPI_Alltoallv(m_send.data(), m_scnt.data(), m_sdspl.data(), MPI_DOUBLE, (double*) m_recv.data(), m_rcnt.data(), m_rdspl.data(), MPI_DOUBLE, m_comm);

    y = 0.0;
    /*
    for (int ifp=0; ifp<m_nprocs; ++ifp)
      {
	for (int i=0; i<m_rcnt[ifp]; ++i)
	  {
	    m_recv[m_rdspl[ifp] + i];
	  }
      }
    */

    for (int i=0; i<m_recv.size(); ++i)
      {
	y[m_alltrueSD[i]] = m_recv[i];
      }

  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    // Given an input vector x of local true SD DOF's, set the output vector y of local true interface DOF's, which may require values from other processes.
    MFEM_VERIFY(y.Size() == m_send.size(), "");

    for (int i=0; i<m_recv.size(); ++i)
      {
	m_recv[i] = x[m_alltrueSD[i]];
      }

    // The roles of receive and send data are reversed.
    MPI_Alltoallv(m_recv.data(), m_rcnt.data(), m_rdspl.data(), MPI_DOUBLE, m_send.data(), m_scnt.data(), m_sdspl.data(), MPI_DOUBLE, m_comm);

    std::vector<int> cnt;
    cnt.assign(m_nprocs, 0);

    for (int i=0; i<y.Size(); ++i) // loop over local true interface DOF's
      {
	y[i] = m_send[m_sdspl[m_iftToSDrank[i]] + cnt[m_iftToSDrank[i]]];
	cnt[m_iftToSDrank[i]]++;
      }

    { // debugging
      for (int i=0; i<m_nprocs; ++i)
	{
	  MFEM_VERIFY(cnt[i] == m_scnt[i], "");
	}
    }
    
    /*
    //gf = 0.0;
    for (int i=0; i<fullWidth; ++i)
      {
      	if (id[i] >= 0)
	  gf[i] = x[id[i]];
      }
    
    gf.GetTrueDofs(y);
    */
  }
};



#define DDMCOMPLEX

class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_, ParFiniteElementSpace *fespaceGlobal, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
		       const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
		       std::vector<int> *interfaceLocalIndex_, const double k2_);

  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    // x and y are vectors of true DOF's on the subdomain interfaces and exterior boundary. 
    // Degrees of freedom in x and y are ordered as follows: x = [x_0, x_1, ..., x_{N-1}];
    // N = numSubdomains, and on subdomain m, x_m = [u_m^s, f_m, \rho_m];
    // u_m^s is the vector of true DOF's of u on the entire surface of subdomain m, for a field u in a Nedelec space on subdomain m;
    // f_m = [f_{m,0}, ..., f_{m,p-1}] is an auxiliary vector of true DOF's in Nedelec spaces on all p interfaces on subdomain m;
    // \rho_m = [\rho_{m,0}, ..., \rho_{m,p-1}] is an auxiliary vector of true DOF's in H^1 (actually H^{1/2}) FE spaces on all
    // p interfaces on subdomain m.
    
    // The surface of subdomain m equals the union of subdomain interfaces and a subset of the exterior boundary.
    // There are redundant DOF's for f and \rho at subdomain corner edges (intersections of interfaces), i.e. discontinuity on corners.
    // The surface bilinear forms and their matrices are defined on subdomain interfaces, not the entire subdomain boundary.
    // The surface DOF's for a subdomain are indexed according to the entire subdomain mesh boundary, and we must use maps between
    // those surface DOF's and DOF's on the individual interfaces.
    
    globalOp->Mult(x, y);

    //BlockOperator *globalSubdomainOp
    //m_globalSubdomainOp->Mult(x, y);
    //globalInterfaceOp->Mult(x, y);
  }  

#ifndef DDMCOMPLEX
  void FullSystemAnalyticTest()
  {
    Vector uSD, uIF, ubdry, ufullSD, AufullSD, diffIF;

    diffIF.SetSize(ifespace[0]->GetTrueVSize());

    VectorFunctionCoefficient E(3, test2_E_exact);
    VectorFunctionCoefficient y(3, test2_RHS_exact);
    VectorFunctionCoefficient f0(3, test2_f_exact_0);
    VectorFunctionCoefficient f1(3, test2_f_exact_1);
    FunctionCoefficient rho0(test2_rho_exact_0);
    FunctionCoefficient rho1(test2_rho_exact_1);

    int fullSize = 0;
    int ifSize = 0;

    std::vector<int> sdFullSize(numSubdomains);
    
    for (int m=0; m<numSubdomains; ++m)
      {
	int fullSize_m = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    fullSize_m += fespace[m]->GetTrueVSize();
	    ifSize += tdofsBdry[m].size();
	  }

	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    fullSize_m += ifespace[interfaceIndex]->GetTrueVSize();
	    fullSize_m += iH1fespace[interfaceIndex]->GetTrueVSize();

	    ifSize += ifespace[interfaceIndex]->GetTrueVSize();
	    ifSize += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	fullSize += fullSize_m;
	sdFullSize[m] = fullSize_m;
      }

    Vector xfull(fullSize);
    xfull = 0.0;
    Vector Axfull(fullSize);
    Axfull = 0.0;
    Vector yfull(fullSize);
    yfull = 0.0;

    Vector xif(ifSize);
    Vector Cxif(ifSize);
    xif = 0.0;

    int osFull = 0;

    for (int m=0; m<numSubdomains; ++m)
      {
	ufullSD.SetSize(sdFullSize[m]);
	AufullSD.SetSize(sdFullSize[m]);

	ufullSD = 0.0;

	int osFullSD = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    uSD.SetSize(fespace[m]->GetTrueVSize());
	    ubdry.SetSize(tdofsBdry[m].size());

	    osFullSD += fespace[m]->GetTrueVSize();
	    
	    ParGridFunction x(fespace[m]);
	    x.ProjectCoefficient(E);
	    x.GetTrueDofs(uSD);
	    
	    tdofsBdryInjectionTranspose[m]->Mult(uSD, ubdry);

	    cout << m_rank << ": ubdry norm on SD " << m << " = " << ubdry.Norml2() << endl;

	    /*
	    { // Subtest, to check u restricted to the interface from each subdomain.
	      uIF.SetSize(ifespace[0]->GetTrueVSize());
	      InterfaceToSurfaceInjection[m][0]->MultTranspose(uSD, uIF);

	      if (m == 0)
		{
		  diffIF = uIF;

		  cout << m_rank << ": uIF[0] norm " << diffIF.Norml2() << endl;
		}
	      else
		{
		  ParGridFunction Eif(ifespace[0]);
		  Eif.ProjectCoefficient(E);
		  uIF = 0.0;
		  Eif.GetTrueDofs(uIF);
		  
		  diffIF -= uIF;
		  cout << m_rank << ": diffIF norm " << diffIF.Norml2() << endl;
		}
	    }
	    */
	    
	    for (int i=0; i<tdofsBdry[m].size(); ++i)
	      {
		xif[block_trueOffsets[m] + i] = ubdry[i];
	      }

	    for (int i=0; i<uSD.Size(); ++i)
	      {
		ufullSD[i] = uSD[i];
	      }

	    x.ProjectCoefficient(y);
	    x.GetTrueDofs(uSD);
	    
	    for (int i=0; i<uSD.Size(); ++i)
	      {
		yfull[osFull + i] = uSD[i];
	      }
	  }

	int osF = 0;
	
	int os = tdofsBdry[m].size();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    uIF.SetSize(ifespace[interfaceIndex]->GetTrueVSize());

	    ParGridFunction x(ifespace[interfaceIndex]);
	    if (m == 0)
	      x.ProjectCoefficient(f0);
	    else
	      x.ProjectCoefficient(f1);
	    
	    x.GetTrueDofs(uIF);

	    for (int j=0; j<uIF.Size(); ++j)
	      {
		xif[block_trueOffsets[m] + os + j] = uIF[j];
		ufullSD[osFullSD + j] = uIF[j];
	      }
	    
	    os += ifespace[interfaceIndex]->GetTrueVSize();

	    osF = osFullSD;
	    osFullSD += ifespace[interfaceIndex]->GetTrueVSize();

	    uIF.SetSize(iH1fespace[interfaceIndex]->GetTrueVSize());
	    
	    ParGridFunction r(iH1fespace[interfaceIndex]);
	    if (m == 0)
	      r.ProjectCoefficient(rho0);
	    else
	      r.ProjectCoefficient(rho1);
	    
	    r.GetTrueDofs(uIF);

	    for (int j=0; j<uIF.Size(); ++j)
	      {
		xif[block_trueOffsets[m] + os + j] = uIF[j];
		ufullSD[osFullSD + j] = uIF[j];
	      }
	    
	    os += iH1fespace[interfaceIndex]->GetTrueVSize();

	    osFullSD += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(sdFullSize[m] == osFullSD, "");

	Asd[m]->Mult(ufullSD, AufullSD);

	double normAuF = 0.0;
	for (int j=0; j<ifespace[0]->GetTrueVSize(); ++j)
	  normAuF += AufullSD[osF + j] * AufullSD[osF + j];

	normAuF = sqrt(normAuF);

	cout << m_rank << ": normAuF " << normAuF << " on sd " << m << endl;
	
	for (int j=0; j<sdFullSize[m]; ++j)
	  {
	    xfull[osFull + j] = ufullSD[j];
	    Axfull[osFull + j] = AufullSD[j];
	  }
	
	osFull += sdFullSize[m];
      }

    MFEM_VERIFY(osFull == fullSize, "");

    globalInterfaceOp->Mult(xif, Cxif);
    //Cxif = 0.0;  // TODO: remove
    
    double Cxif_norm = Cxif.Norml2();
    cout << m_rank << ": Cxif norm " << Cxif_norm << ", Axfull norm " << Axfull.Norml2() << endl;
    
    double Cxif_us = 0.0;
    
    osFull = 0;
    
    for (int m=0; m<numSubdomains; ++m)
      {
	int osFullSD = 0;

	if (pmeshSD[m] != NULL)
	  {
	    uSD.SetSize(fespace[m]->GetTrueVSize());
	    ubdry.SetSize(tdofsBdry[m].size());
    
	    for (int i=0; i<tdofsBdry[m].size(); ++i)
	      {
		ubdry[i] = Cxif[block_trueOffsets[m] + i];
	      }

	    double Cxif_us_m = ubdry.Norml2();
	    Cxif_us += Cxif_us_m * Cxif_us_m;
	    
	    tdofsBdryInjection[m]->Mult(ubdry, uSD);

	    for (int i=0; i<uSD.Size(); ++i)
	      {
		Axfull[osFull + i] += uSD[i];
	      }

	    osFullSD += fespace[m]->GetTrueVSize();
	  }

	int os = tdofsBdry[m].size();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    for (int j=0; j<ifespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		Axfull[osFull + osFullSD + j] += Cxif[block_trueOffsets[m] + os + j];
	      }

	    os += ifespace[interfaceIndex]->GetTrueVSize();
	    os += iH1fespace[interfaceIndex]->GetTrueVSize();

	    osFullSD += ifespace[interfaceIndex]->GetTrueVSize();
	    osFullSD += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(sdFullSize[m] == osFullSD, "");
	
	osFull += sdFullSize[m];
      }

    Cxif_us = sqrt(Cxif_us);
    
    cout << m_rank << ": Cxif u_s component norm " << Cxif_us << ", f and rho component norm " << Cxif_norm - Cxif_us << endl;

    // Now Axfull is (A + C) xfull, which should equal yfull.
    Vector dfull(fullSize);
    Vector dfull_u(fullSize);
    Vector dfull_us(fullSize);
    Vector dfull_f(fullSize);
    Vector dfull_rho(fullSize);
    Vector Axfull_u(fullSize);
    Vector Axfull_f(fullSize);
    Vector Axfull_rho(fullSize);

    dfull_u = 0.0;
    dfull_us = 0.0;
    dfull_f = 0.0;
    dfull_rho = 0.0;

    Axfull_u = 0.0;
    Axfull_f = 0.0;
    Axfull_rho = 0.0;
    
    for (int i=0; i<fullSize; ++i)
      dfull[i] = Axfull[i] - yfull[i];

    osFull = 0;
    for (int m=0; m<numSubdomains; ++m)
      {
	int osm = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    for (int i=0; i<fespace[m]->GetTrueVSize(); ++i)
	      {
		dfull_u[osFull + i] = Axfull[osFull + i] - yfull[osFull + i];
		Axfull_u[osFull + i] = Axfull[osFull + i];
	      }

	    osm += fespace[m]->GetTrueVSize();
	    
	    for (set<int>::const_iterator it = tdofsBdry[m].begin(); it != tdofsBdry[m].end(); ++it)
	      {
		dfull_us[osFull + (*it)] = dfull_u[osFull + (*it)];
	      }
	  }

	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    for (int j=0; j<ifespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		dfull_f[osFull + osm + i] = Axfull[osFull + osm + i] - yfull[osFull + osm + i];
		Axfull_f[osFull + osm + i] = Axfull[osFull + osm + i];
	      }

	    osm += ifespace[interfaceIndex]->GetTrueVSize();

	    for (int j=0; j<iH1fespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		dfull_rho[osFull + osm + i] = Axfull[osFull + osm + i] - yfull[osFull + osm + i];
		Axfull_rho[osFull + osm + i] = Axfull[osFull + osm + i];
	      }
	    
	    osm += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(osm == sdFullSize[m], "");
	osFull += sdFullSize[m];
      }
    
    cout << m_rank << ": dfull norm " << dfull.Norml2() << ", y norm " << yfull.Norml2() << ", (A+C)x norm " << Axfull.Norml2() << endl;
    cout << m_rank << ": dfull_u norm " << dfull_u.Norml2() << ", (A+C)x_u norm " << Axfull_u.Norml2() << endl;
    cout << m_rank << ": dfull_us norm " << dfull_us.Norml2() << endl;
    cout << m_rank << ": dfull_f norm " << dfull_f.Norml2() << ", (A+C)x_f norm " << Axfull_f.Norml2() << endl;
    cout << m_rank << ": dfull_rho norm " << dfull_rho.Norml2() << ", (A+C)x_rho norm " << Axfull_rho.Norml2() << endl;
  }
#endif

  void GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobalRe, Vector & sourceGlobalIm, Vector & sourceReduced) const;

#ifdef DDMCOMPLEX
  void PrintSubdomainError(const int sd, Vector & u, Vector & eSD)
  {
    if (fespace[sd] == NULL)
      return;
    
    ParGridFunction x(fespace[sd]);
    VectorFunctionCoefficient E(3, test2_E_exact);
    //x.ProjectCoefficient(E);

    Vector uSD(fespace[sd]->GetTrueVSize());

    Vector zeroVec(3);
    zeroVec = 0.0;
    VectorConstantCoefficient vzero(zeroVec);

    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      uSD[i] = u[i];

    x.SetFromTrueDofs(uSD);

    double errRe = x.ComputeL2Error(E);
    double normXRe = x.ComputeL2Error(vzero);

    DataCollection *dc = NULL;
    const bool visit = true;
    if (visit && sd == 1)
      {
	//x.ProjectCoefficient(E);

       bool binary = false;
       if (binary)
	 {
#ifdef MFEM_USE_SIDRE
	   dc = new SidreDataCollection("ddsol", pmeshSD[sd]);
#else
	   MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
	 }
       else
	 {
	   dc = new VisItDataCollection("usd1", pmeshSD[sd]);
	   dc->SetPrecision(8);
	   // To save the mesh using MFEM's parallel mesh format:
	   // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
	 }
       dc->RegisterField("solution", &x);
       dc->SetCycle(0);
       dc->SetTime(0.0);
       dc->Save();
     }

    // Overwrite real part with imaginary part, only for the subdomain true DOF's.
    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      uSD[i] = u[block_ComplexOffsetsSD[sd][1] + i];

    MFEM_VERIFY(u.Size() == block_ComplexOffsetsSD[sd][2], "");
    
    x.SetFromTrueDofs(uSD);
    const double errIm = x.ComputeL2Error(vzero);
    //const double errIm = x.ComputeL2Error(E);

    ParGridFunction zerogf(fespace[sd]);
    zerogf = 0.0;
    const double normE = zerogf.ComputeL2Error(E);

    const double relErrRe = errRe / normE;
    const double relErrTot = sqrt((errRe*errRe) + (errIm*errIm)) / normE;

    /*
    double relErrReMax = -1.0;
    double relErrTotMax = -1.0;
    
    MPI_Allreduce(&relErrRe, &relErrReMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&relErrTot, &relErrTotMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    */
    
    //if (m_rank == 0)
      {
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Re = " << errRe << endl;
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Im = " << errIm << endl;
	cout << m_rank << ": sd " << sd << " || E_h ||_{L^2} Re = " << normXRe << endl;
	cout << m_rank << ": sd " << sd << " || E ||_{L^2} Re = " << normE << endl;
	cout << m_rank << ": sd " << sd << " rel err Re " << relErrRe << endl; // ", max " << relErrReMax << endl;
	cout << m_rank << ": sd " << sd << " rel err tot " << relErrTot << endl; // ", max " << relErrTotMax << endl;

	eSD[0] = errRe;
	eSD[1] = errIm;
	eSD[2] = normE;
      }
  }
#endif

  void RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced, Vector & solDomain);
    
private:

  int m_rank;

  mutable bool testing;
  
  const double k2;

  bool realPart;
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;

  ParMesh *pmeshGlobal;
  ParMesh **pmeshSD;  // Subdomain meshes
  ParMesh **pmeshIF;  // Interface meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
  ParFiniteElementSpace **fespace, **ifespace, **iH1fespace;
  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad, **ifNDH1gradT, **ifH1mass, **ifND_FS;
  Operator **ifNDmassInv, **ifH1massInv;
  HypreParMatrix **sdND;
  HypreParMatrix **sdNDcopy;
  HypreParMatrix **A_SS;
  HypreParMatrix **sdNDPen;
  HypreParMatrix **sdNDPlusPen;
  SparseMatrix **sdNDPenSp;
  ParBilinearForm **bf_sdND;
  Operator **sdNDinv;
#ifdef DDMCOMPLEX
  Operator **AsdRe, **AsdIm, **AsdP, **invAsdComplex;
#ifdef SPARSE_ASDCOMPLEX
  SparseMatrix **SpAsdComplex;
  HypreParMatrix **HypreAsdComplex;
  HYPRE_Int **SpAsdComplexRowSizes;
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
  std::vector<Array2D<HypreParMatrix*> > AsdRe_HypreBlocks;
  std::vector<Array2D<HypreParMatrix*> > AsdIm_HypreBlocks;
  std::vector<Array2D<double> > AsdRe_HypreBlockCoef;
  std::vector<Array2D<double> > AsdIm_HypreBlockCoef;
  std::vector<MPI_Comm> sd_com;
#endif
#endif
  BlockOperator **AsdComplex;
  BlockDiagonalPreconditioner **precAsdComplex;
#else
  Operator **Asd;
#endif
  Operator **ASPsd;
  Operator **invAsd;
  Solver **precAsd;

  // TODO: it may be possible to eliminate ifND_FS. It is assembled as a convenience, to avoid summing entries input to ASPsd.
  
  Vector **ySD;
  
  BlockOperator **injSD;
  
  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

  std::vector<int> ifNDtrue, ifH1true;
  
#ifdef DDMCOMPLEX
  BlockOperator *globalInterfaceOpRe, *globalInterfaceOpIm;
  std::vector<Array<int> > block_ComplexOffsetsSD;
  Array<int> block_trueOffsets2;  // Offsets used in globalOp
  BlockOperator **injComplexSD;
#endif

  BlockOperator *globalInterfaceOp, *globalSubdomainOp;
  
  Operator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  BlockOperator *NDinv;
  
  vector<set<int> > tdofsBdry;
  SetInjectionOperator **tdofsBdryInjection;
  Operator **tdofsBdryInjectionTranspose;
  
  double alpha, beta, gamma;
  double alphaInverse, betaOverAlpha, gammaOverAlpha;
  double alphaRe, betaRe, gammaRe;
  double alphaIm, betaIm, gammaIm;

  bool preconditionerMode;
  
  std::vector<std::vector<InjectionOperator*> > InterfaceToSurfaceInjection;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionData;
  //std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionFullData;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionGlobalData;

  std::vector<Array<int> > rowTrueOffsetsSD, colTrueOffsetsSD;
#ifdef DDMCOMPLEX
  std::vector<Array<int> > rowTrueOffsetsComplexSD, colTrueOffsetsComplexSD;
  std::vector<Array<int> > rowTrueOffsetsComplexIF, colTrueOffsetsComplexIF;
#endif
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIF, colTrueOffsetsIF;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFL, colTrueOffsetsIFL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFR, colTrueOffsetsIFR;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBL, colTrueOffsetsIFBL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBR, colTrueOffsetsIFBR;
  std::vector<Array<int> > trueOffsetsSD;
  Array<int> rowSROffsetsSD, colSROffsetsSD;

#ifdef EQUATE_REDUNDANT_VARS
  std::vector<Array<int> > rowTrueOffsetsIFRR, colTrueOffsetsIFRR;
#endif
  
  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void SetParameters()
  {
    const double k = sqrt(k2);

    const double cTE = 0.5;  // From RawatLee2010
    const double cTM = 4.0;  // From RawatLee2010
    
    // Note that PengLee2012 recommends cTE = 1.5 * cTM. 

    // TODO: take these parameters from the mesh and finite element space.
    //const double h = 0.0350769;
    //const double h = 0.0175385;
    const double h = 1.4e-1;
    //const double h = 0.5;
    const double feOrder = 1.0;

    const double ktTE = cTE * M_PI * feOrder / h;
    const double ktTM = cTM * M_PI * feOrder / h;

    /*    
    const double ktTE = 3.0 * k; // From PengLee2012
    const double ktTM = 2.0 * k; // From PengLee2012
    */
    
    //const double kzTE = sqrt(8.0 * k2);
    //const double kzTM = sqrt(3.0 * k2);

    const double kzTE = sqrt((ktTE*ktTE) - k2);
    const double kzTM = sqrt((ktTM*ktTM) - k2);

    // PengLee2012
    // alpha = ik
    // beta = i / (k + i kzTE) = i * (k - i kzTE) / (k^2 + kzTE^2) = (kzTE + ik) / (k^2 + kzTE^2)
    // gamma = 1 / (k^2 + i k * kzTM) = (k^2 - i k * kzTM) / (k^4 + k^2 kzTM^2)
	
    // Real part
    alphaRe = 0.0;
    betaRe = kzTE / (k2 + (kzTE * kzTE));
    gammaRe = 1.0 / (k2 + (kzTM * kzTM));

    // Imaginary part
    alphaIm = k;
    betaIm = k / (k2 + (kzTE * kzTE));
    gammaIm = -kzTM / (k * (k2 + (kzTM * kzTM)));


    // RawatLee2010
    // alpha = -ik
    // beta = i / (k + kzTE) = i / (k - i sqrt((k_tau^{max,te})^2 - k^2)) = i / (k - i sqrt(ktTE^2 - k^2))
    //      = i (k + i sqrt(ktTE^2 - k^2)) / [k^2 + ktTE^2 - k^2] = i (k + i sqrt(ktTE^2 - k^2)) / ktTE^2
    //      = [-sqrt(ktTE^2 - k^2) + ik] / ktTE^2 = [-kzTE + ik] / ktTE^2
    // gamma = 1 / [k^2 + k kzTM] = 1 / [k^2 - ik sqrt((k_tau^{max,tm})^2 - k^2)] = 1 / [k^2 - ik sqrt(ktTM^2 - k^2)]
    //       = [k^2 + ik sqrt(ktTM^2 - k^2)] / (k^4 + k^2 ktTM^2 - k^4) = [k^2 + ik sqrt(ktTM^2 - k^2)] / (k^2 ktTM^2)
    //       = [1 + i (1/k) sqrt(ktTM^2 - k^2)] / ktTM^2 = [1 + i (kzTM / k)] / ktTM^2
    
    // Real part
    alphaRe = 0.0;
    betaRe = -kzTE / (ktTE * ktTE);
    gammaRe = 1.0 / (ktTM * ktTM);

    // Imaginary part
    alphaIm = -k;
    betaIm = k / (ktTE * ktTE);
    gammaIm = kzTM / (k * (ktTM * ktTM));

    /*
    {
      //////////// Testing 
      alphaRe = 1.0;
      betaRe = 1.0;
      gammaRe = 1.0;

      alphaIm = 0.0;
      betaIm = 0.0;
      gammaIm = 0.0;
    }
    */
  }

  void SetToRealParameters()
  {
    realPart = true;

    alpha = alphaRe;
    beta = betaRe;
    gamma = gammaRe;

    alphaInverse = alphaRe / ((alphaRe * alphaRe) + (alphaIm * alphaIm));

    betaOverAlpha = ((alphaRe * betaRe) + (alphaIm * betaIm)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
    gammaOverAlpha = ((alphaRe * gammaRe) + (alphaIm * gammaIm)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
  }
  
  void SetToImaginaryParameters()
  {
    realPart = false;

    alpha = alphaIm;
    beta = betaIm;
    gamma = gammaIm;

    alphaInverse = -alphaIm / ((alphaRe * alphaRe) + (alphaIm * alphaIm));

    betaOverAlpha = ((alphaRe * betaIm) - (alphaIm * betaRe)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
    gammaOverAlpha = ((alphaRe * gammaIm) - (alphaIm * gammaRe)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
  }
  
  void SetToPreconditionerParameters()
  {
    realPart = true;

    /*
    alpha = 1.0;
    beta = 1.0;
    gamma = 1.0;
    */
    
    alpha = sqrt((alphaRe * alphaRe) + (alphaIm * alphaIm));
    beta = sqrt((betaRe * betaRe) + (betaIm * betaIm));
    gamma = sqrt((gammaRe * gammaRe) + (gammaIm * gammaIm));

    alphaInverse = 1.0 / alpha;

    betaOverAlpha = beta / alpha;
    gammaOverAlpha = gamma / alpha;
  }

  void CreateInterfaceMatrices(const int interfaceIndex);
  
  // Create operator C_{sd0,sd1} in the block space corresponding to [u_m^s, f_i, \rho_i]. Note that the u_m^I blocks are omitted (just zeros).
  Operator* CreateCij(const int localInterfaceIndex, const int orientation);

  // Create operator C_{sd0,sd1} R_{sd1}^T. The operator returned here is of size n_{sd0} by n_{sd1}, where n_{sd} is the sum of
  // tdofsBdry[sd].size() and ifespace[interfaceIndex]->GetTrueVSize() and iH1fespace[interfaceIndex]->GetTrueVSize() for all interfaces of subdomain sd.
  Operator* CreateInterfaceOperator(const int localInterfaceIndex, const int orientation);

  void CreateSubdomainMatrices(const int subdomain);

  STRUMPACKSolver* CreateStrumpackSolver(Operator *Arow, MPI_Comm comm)
  {
    //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, comm);
    STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, comm);
    strumpack->SetPrintFactorStatistics(true);
    strumpack->SetPrintSolveStatistics(false);
    strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
    strumpack->SetOperator(*Arow);
    strumpack->SetFromCommandLine();
    return strumpack;
  }
  
  Solver* CreateSubdomainPreconditionerStrumpack(const int subdomain);

  void SetOffsetsSD(const int subdomain);
  
  //#define SCHURCOMPSD
  
  // Create operator A_m for subdomain m, in the block space corresponding to [u_m, f_m^s, \rho_m^s].
  // We use mappings between interface and subdomain boundary DOF's, so there is no need for interior and surface blocks on each subdomain.
  Operator* CreateSubdomainOperator(const int subdomain);
  
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
  void CreateSubdomainHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block, Array2D<double>& blockCoefficient);
#endif
  
  // This is the same operator as CreateSubdomainOperator, except it is stored as a strumpack matrix rather than a block operator. 
  Operator* CreateSubdomainOperatorStrumpack(const int subdomain);
    
};
  
#endif  // DDOPER_HPP