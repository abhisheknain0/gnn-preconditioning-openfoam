/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "pimpleSingleRegionControl.H"
#include "setDeltaT.H"
#include "iterReduceIncompressibleFluid.H"
using namespace Foam;

int main(int argc, char* argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    solvers::iterReduceIncompressibleFluid solver(mesh);
    pimpleSingleRegionControl pimple(solver.pimple);

    setDeltaT(runTime, solver);

    Info<< nl << "Starting time loop" << nl << endl;

    while (pimple.run(runTime))
    {
        pimple.read();

        solver.preSolve();

        adjustDeltaT(runTime, solver);
        runTime++;

        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        while (pimple.loop())
        {
            solver.moveMesh();
            solver.motionCorrector();
            solver.fvModels().correct();
            solver.prePredictor();
            if (pimple.firstIter())
            {
                solver.injectPredictorFields();
            }
            solver.momentumPredictor();
            solver.thermophysicalPredictor();
            solver.pressureCorrector();
            solver.postCorrector();
        }

        solver.postSolve();
        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End" << endl;
    return 0;
}

// ************************************************************************* //
