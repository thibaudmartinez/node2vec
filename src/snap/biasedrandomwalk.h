#ifndef RAND_WALK_H
#define RAND_WALK_H

#include "network.h"
#include "graph.h"

typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

/**
 * Preprocesses transition probabilities for random walks.
 * Has to be called once before SimulateWalk calls.
 */
void PreprocessTransitionProbs(
        PWNet &InNet,
        const double &ParamP,
        const double &ParamQ,
        const bool &Verbose
);

/**
 * Simulates one walk and writes it into Walk vector.
 */
void SimulateOneWalk(
        PWNet &InNet,
        int64 StartNId,
        const int &WalkLen,
        TRnd &Rnd,
        TIntV &WalkV
);

/**
 * Simulates multiple walks.
 */
const TVVec<TInt, int64> SimulateWalks(
        PWNet &InNet,
        const int &WalkLen,
        const int &NumWalks,
        const double &ParamP,
        const double &ParamQ,
        const bool &Verbose
);

const TVVec<TInt, int64> SimulateWalks(
        PWNet &InNet,
        const int &WalkLen,
        const int &NumWalks,
        const double &ParamP,
        const double &ParamQ,
        const bool &Verbose,
        const int &RandSeed
);

/**
 * Predicts approximate memory required for preprocessing the graph.
 */
int64 PredictMemoryRequirements(PWNet &InNet);

#endif //RAND_WALK_H
