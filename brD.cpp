//Laplace's equation with different boundary conditions using 
//MFEM library - Parallel Version.

//Compile with: make brD

//Sample runs: mpirun -np 4 brD

//Description: This code his code demonstrates the use of MFEM to define a
//              simple finite element discretization of the Laplace problem
//              -Div(rho(x)Delta u) = f with a variety of boundary conditions.

//             Specifically, we discretize using a FE space of the specified
//             order using a continuous space. We then apply
//             Dirichlet, Neumann homogeneous and inhomogeneous respectively,
//             and Periodic boundary conditions on different portions of a
//             predefined mesh.

//             The predefined mesh consists of a rectangle with two holes
//             removed (see below). The narrow ends of the mesh are connected
//             to form a Periodic boundary condition. The lower edge and the 
//             upper edge (tagged with attribute 1 and attribute 2) receives 
//             an inhomogeneous Neumann boundary condition.  The circular 
//             hole on the left (attribute 3) enforces a Dirichlet boundary 
//             condition. Finally, a natural boundary condition, or homogeneous 
//             Neumann BC, is applied to the circular hole on the right (attribute 4).

//                    Attribute 3    ^ y  Attribute 2
//                          \        |      /
//                       +-----------+-----------+
//                       |    \_     |     _     |
//                       |    / \    |    / \    |
//                    <--+---+---+---+---+---+---+--> x
//                       |    \_/    |    \_/    |
//                       |           |      \    |
//                       +-----------+-----------+       (hole radii are
//                            /      |        \            adjustable)
//                    Attribute 1    v    Attribute 4
//            The boundary conditions are defined as (where u is the solution
//            field):
//                  Dirichlet: u = d
//                  Neumann: n.Grad(u) = g
//            Remark: We take as reference example 27 of the parallel version of MFEM
//            library.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double a_ = 0.2; // Predetermined value of the radii of the hole.

//Normal to hole with boundary attribute 4
void n4Vec(const Vector &x, Vector &n) { n = x; n[0] -= 0.5; n /= -n.Norml2(); }

double myFun1(const Vector &x); //Function rho.

double funCoef(const Vector &x); //Rho coefficient.

double g1Nbc(const Vector &x); //Function g1 inhomogeneous Neumann BC.

double g2Nbc(const Vector &x); //Function g2 inhomogeneous Neumann BC.

double g3Dbc(const Vector &x); //Function g3 Dirichlet BC.

double g4Nbc(const Vector &x); //Function g4 inhomogeneous Neumann BC.

double f_analytic(const Vector & x); //-Div(rho(x)Delta u) = f.

double usol(const Vector & x); //Solution function.

void u_grad_exact(const Vector &x, Vector &usol);//Solution function gradient.

Mesh * GenerateSerialMesh(int ref);

int main(int argc, char *argv[])
{
    //1. Initialize MPI and HYPRE.
    Mpi::Init();
    if (!Mpi::Root()){mfem::out.Disable(); mfem::err.Disable();}
    Hypre::Init();

    //2. Parse command-line options.
    int ser_ref_levels = 2;
    int par_ref_levels = 4;
    int order = 1;
    bool visualization = true;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&a_, "-a", "--radius",
                  "Radius of holes in the mesh.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(mfem::out);
        return 1;
    }
    if (a_ < 0.01) 
   {
      mfem::out << "Hole radius too small, resetting to 0.01.\n";
      a_ = 0.01;
   }
   if (a_ > 0.49)
   {
      mfem::out << "Hole radius too large, resetting to 0.49.\n";
      a_ = 0.49;
   }
   // 3. Construct the serial mesh and refine it if requested.
   Mesh *mesh = GenerateSerialMesh(ser_ref_levels);
   int dim = mesh->Dimension();
   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }
   // 5. Define a parallel finite element space on the parallel mesh. 
   //    Here we use continuous Lagrange finite elements of the
   //    specified order.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   mfem::out << "Number of finite element unknowns" << size << endl;

   // 6. Create "marker arrays" to define the portions of boundary associated
   //    with each type of boundary condition. These arrays have an entry
   //    corresponding to each boundary attribute. Placing a '1' in entry i
   //    marks attribute i+1 as being active, '0' is inactive.

   Array<int> nbc_bdr1(pmesh.bdr_attributes.Max());
   Array<int> nbc_bdr2(pmesh.bdr_attributes.Max());
   Array<int> dbc_bdr3(pmesh.bdr_attributes.Max());
   Array<int> nbc_bdr4(pmesh.bdr_attributes.Max());

   nbc_bdr1 = 0; nbc_bdr1[0] = 1;
   nbc_bdr2 = 0; nbc_bdr2[1] = 1;
   dbc_bdr3 = 0; dbc_bdr3[2] = 1;
   nbc_bdr4 = 0; nbc_bdr4[3] = 1;

   // The linear system must be modifed to enforce an essential
   // Dirichlet boundary condition.
   Array<int> ess_tdof_list(0);
   fespace.GetEssentialTrueDofs(dbc_bdr3, ess_tdof_list);

   //7. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions.
   FunctionCoefficient p(funCoef);
   ParGridFunction rho1(&fespace); //Let's add the coefficient rho
   rho1.ProjectCoefficient(p);
   GridFunctionCoefficient rho(&rho1);
   // Now we convert the boundary conditions and the f in the Laplace's
   // equation in to a FunctionCoeffcient. 
   FunctionCoefficient g1(g1Nbc);
   FunctionCoefficient g2(g2Nbc);
   FunctionCoefficient g3(g3Dbc);
   FunctionCoefficient g4(g4Nbc);
   FunctionCoefficient f_an(f_analytic);

   // Since the n.Grad(u) terms arise by integrating -Div(rho Grad(u)) by parts we
   // must introduce the coefficient 'rho' (rho) into the boundary conditions.
   // Therefore, in the case of the Neumann BC, we actually enforce rho n.Grad(u)
   // = rho g rather than simply n.Grad(u) = g.
   ProductCoefficient m_nbc1Coef(g1, rho);
   ProductCoefficient m_nbc2Coef(g2, rho);
   ProductCoefficient m_nbc4Coef(g4, rho);

   // 8. Define the solution vector u as a parallel finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of zero.
   ParGridFunction u(&fespace);
   u = 0.0;

   // 9. To study the solution we convert usol to grid function and the 
   //solution gradien function, the idea is to calculate the 
   //error between the MFEM solution and the solution.
   FunctionCoefficient u1(usol);
   ParGridFunction uSol(&fespace);
   uSol.ProjectCoefficient(u1);
   VectorFunctionCoefficient u_grad(dim, u_grad_exact);

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(rho);
   a.AddDomainIntegrator(integ);
   a.Assemble();

   // 11. Assemble the parallel linear form for the right hand side vector.
   // Set the Dirichlet values in the solution vector
   ParLinearForm b(&fespace);
   u.ProjectBdrCoefficient(g3,dbc_bdr3);
   //Add the f parameter to the vector b
   b.AddDomainIntegrator(new DomainLFIntegrator(f_an));
   //Add the desired value for n.Grad(u) on the Neumann Boundary 1
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbc1Coef),nbc_bdr1);
   //Add the desired value for n.Grad(u) on the Neumann Boundary 2
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbc2Coef),nbc_bdr2);
   //Add the desired valuer for n.Grad(u) on the Neumann Boundary 3
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbc4Coef),nbc_bdr4);
   b.Assemble();
    // 12. Construct the linear system.
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

    // 13. Solver the linear system AX=B.
    HypreSolver *amg = new HypreBoomerAMG;
    HyprePCG pcg(MPI_COMM_WORLD);
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(200);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(*amg);
    pcg.SetOperator(*A);
    pcg.Mult(B, X);
    delete amg;

    // 14. Recover the parallel grid function corresponding to U. This is the
    //     local finite element solution on each processor.
    a.RecoverFEMSolution(X, b, u);
    // 14.1 Compute the H^1 norms of the error.
    double h1_err_prev = 0.0;
    double h_prev = 0.0;
    double h1_err = u.ComputeH1Error(&u1,&u_grad,&p,1.0,1.0);
    mfem::out <<"Calculate of the error: " 
             << h1_err << endl;




    // 16. Save the refined mesh and the solution in parallel. This output can be
    //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
    {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }
   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string title_str = "H1";
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u
               << "window_title '" << title_str << " Solution'"
               << " keys 'mmc'" << flush;
      socketstream exact_sol(vishost, visport);
      exact_sol<< "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      exact_sol.precision(8);
      //uSol = uSol.Add(-1.0,u);
      exact_sol << "solution \n" << pmesh << uSol
                << "window_title 'Exact Solution'\n"
                << "keys 'mmc'"<< flush;
      
   }  
   // 18. Free the used memory.
   delete fec;

   return 0; 
}
double myFun1(const Vector &x)
{
   return 1.0/exp(x[0]+x[1]);
}
double funCoef(const Vector &x)
{
   return (myFun1(x));
}
//We are going to define the functions in order to get the boundary condition
//for the function u = cos(pi*x)exp(y)
double g1Nbc(const Vector &x)
{
    return -exp(x[1])*cos(2.0*M_PI*x[0]);
}
double g2Nbc(const Vector &x)
{
    return(-g1Nbc(x));
}
double g3Dbc(const Vector &x)
{
    return cos(2.0*M_PI*x[0])*exp(x[1]);
}
double g4Nbc(const Vector &x)
{
    return -(exp(x[1])*(2.0*x[1]*cos(2.0*M_PI*x[0])-2.0*M_PI*sin(2.0*M_PI*x[0])*(2.0*x[0]-1.0)))/sqrt((2.0*x[0]-1)*(2.0*x[0]-1)+4.0*x[1]*x[1]);
}
// now, we do the the f function
double f_analytic(const Vector & x)
{
    return 2.0*M_PI*exp(-x[0])*(2.0*M_PI*cos(2.0*M_PI*x[0])-sin(2.0*M_PI*x[0]));
}

double usol(const Vector & x)
{
    return cos(2.0*M_PI*x[0])*exp(x[1]);
}

void u_grad_exact(const Vector &x, Vector &usol)
{
   usol[0] = -2.0*M_PI*sin(2.0*M_PI*x[0])*exp(x[1]);
   usol[1] = exp(x[1])*cos(2.0*M_PI*x[0]);
}
void quad_trans(double u, double v, double &x, double &y, bool log = false)
{
   double a = a_; // Radius of disc

   double d = 4.0 * a * (M_SQRT2 - 2.0 * a) * (1.0 - 2.0 * v);

   double v0 = (1.0 + M_SQRT2) * (M_SQRT2 * a - 2.0 * v) *
               ((4.0 - 3 * M_SQRT2) * a +
                (8.0 * (M_SQRT2 - 1.0) * a - 2.0) * v) / d;

   double r = 2.0 * ((M_SQRT2 - 1.0) * a * a * (1.0 - 4.0 *v) +
                     2.0 * (1.0 + M_SQRT2 *
                            (1.0 + 2.0 * (2.0 * a - M_SQRT2 - 1.0) * a)) * v * v
                    ) / d;

   double t = asin(v / r) * u / v;
   if (log)
   {
      mfem::out << "u, v, r, v0, t "
                << u << " " << v << " " << r << " " << v0 << " " << t
                << endl;
   }
   x = r * sin(t);
   y = r * cos(t) - v0;
}

void trans(const Vector &u, Vector &x)
{
   double tol = 1e-4;

   if (u[1] > 0.5 - tol || u[1] < -0.5 + tol)
   {
      x = u;
      return;
   }
   if (u[0] > 1.0 - tol || u[0] < -1.0 + tol || fabs(u[0]) < tol)
   {
      x = u;
      return;
   }

   if (u[0] > 0.0)
   {
      if (u[1] > fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, u[1], x[0], x[1]);
         x[0] += 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, -u[1], x[0], x[1]);
         x[0] += 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] - 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] - 0.5, x[1], x[0]);
         x[0] += 0.5;
         return;
      }
      if (u[0] - 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], 0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] += 0.5;
         return;
      }
   }
   else
   {
      if (u[1] > fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, u[1], x[0], x[1]);
         x[0] -= 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, -u[1], x[0], x[1]);
         x[0] -= 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] + 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] + 0.5, x[1], x[0]);
         x[0] -= 0.5;
         return;
      }
      if (u[0] + 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], -0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] -= 0.5;
         return;
      }
   }
   x = u;
}

Mesh * GenerateSerialMesh(int ref)
{
   Mesh * mesh = new Mesh(2, 29, 16, 24, 2);

   int vi[4];

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o + 0; vi[1] = o + 3; vi[2] = o + 4; vi[3] = o + 1;
      mesh->AddQuad(vi);

      vi[0] = o + 1; vi[1] = o + 4; vi[2] = o + 5; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 5; vi[1] = o + 8; vi[2] = o + 9; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 8; vi[1] = o + 12; vi[2] = o + 15; vi[3] = o + 9;
      mesh->AddQuad(vi);

      vi[0] = o + 11; vi[1] = o + 14; vi[2] = o + 15; vi[3] = o + 12;
      mesh->AddQuad(vi);

      vi[0] = o + 10; vi[1] = o + 13; vi[2] = o + 14; vi[3] = o + 11;
      mesh->AddQuad(vi);

      vi[0] = o + 6; vi[1] = o + 13; vi[2] = o + 10; vi[3] = o + 7;
      mesh->AddQuad(vi);

      vi[0] = o + 0; vi[1] = o + 6; vi[2] = o + 7; vi[3] = o + 3;
      mesh->AddQuad(vi);
   }

   vi[0] =  0; vi[1] =  6; mesh->AddBdrSegment(vi, 1);
   vi[0] =  6; vi[1] = 13; mesh->AddBdrSegment(vi, 1);
   vi[0] = 13; vi[1] = 19; mesh->AddBdrSegment(vi, 1);
   vi[0] = 19; vi[1] = 26; mesh->AddBdrSegment(vi, 1);

   vi[0] = 28; vi[1] = 22; mesh->AddBdrSegment(vi, 2);
   vi[0] = 22; vi[1] = 15; mesh->AddBdrSegment(vi, 2);
   vi[0] = 15; vi[1] =  9; mesh->AddBdrSegment(vi, 2);
   vi[0] =  9; vi[1] =  2; mesh->AddBdrSegment(vi, 2);

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o +  7; vi[1] = o +  3; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 10; vi[1] = o +  7; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 11; vi[1] = o + 10; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 12; vi[1] = o + 11; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  8; vi[1] = o + 12; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  5; vi[1] = o +  8; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  4; vi[1] = o +  5; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  3; vi[1] = o +  4; mesh->AddBdrSegment(vi, 3 + i);
   }

   double d[2];
   double a = a_ / M_SQRT2;

   d[0] = -1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] = -0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.5; mesh->AddVertex(d);

   mesh->FinalizeTopology();

   mesh->SetCurvature(1, true);

   // Stitch the ends of the stack together
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - 3; i++)
      {
         v2v[i] = i;
      }
      // identify vertices on the narrow ends of the rectangle
      v2v[v2v.Size() - 3] = 0;
      v2v[v2v.Size() - 2] = 1;
      v2v[v2v.Size() - 1] = 2;

      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }
   mesh->SetCurvature(3, true);

   for (int l = 0; l < ref; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->Transform(trans);

   return mesh;
}




