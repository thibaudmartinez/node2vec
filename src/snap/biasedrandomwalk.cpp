#include <string>
#include <iostream>
#include <iomanip>

#include "biasedrandomwalk.h"

/**
 * Preprocess alias sampling method
 */
void GetNodeAlias(TFltV &PTblV, TIntVFltVPr &NTTable) {
    int64 N = PTblV.Len();

    TIntV &KTbl = NTTable.Val1;
    TFltV &UTbl = NTTable.Val2;
    for (int64 i = 0; i < N; i++) {
        KTbl[i] = 0;
        UTbl[i] = 0;
    }

    TIntV UnderV;
    TIntV OverV;
    for (int64 i = 0; i < N; i++) {
        UTbl[i] = PTblV[i] * N;
        if (UTbl[i] < 1) {
            UnderV.Add(i);
        } else {
            OverV.Add(i);
        }
    }

    while (UnderV.Len() > 0 && OverV.Len() > 0) {
        int64 Small = UnderV.Last();
        int64 Large = OverV.Last();
        UnderV.DelLast();
        OverV.DelLast();
        KTbl[Small] = Large;
        UTbl[Large] = UTbl[Large] + UTbl[Small] - 1;
        if (UTbl[Large] < 1) {
            UnderV.Add(Large);
        } else {
            OverV.Add(Large);
        }
    }

    while (UnderV.Len() > 0) {
        int64 curr = UnderV.Last();
        UnderV.DelLast();
        UTbl[curr] = 1;
    }

    while (OverV.Len() > 0) {
        int64 curr = OverV.Last();
        OverV.DelLast();
        UTbl[curr] = 1;
    }
}

/**
 * Get random element using alias sampling method.
 */
int64 AliasDrawInt(TIntVFltVPr &NTTable, TRnd &Rnd) {
    int64 N = NTTable.GetVal1().Len();
    TInt X = static_cast<int64>(Rnd.GetUniDev() * N);
    double Y = Rnd.GetUniDev();

    return Y < NTTable.GetVal2()[X] ? X : NTTable.GetVal1()[X];
}

void PrintProgress(std::string Message, double Progress) {
    std::cout << "\r" << Message << std::setprecision(4) << Progress * 100 << "%" << std::flush;
}

void PreprocessNode(
        PWNet &InNet,
        const double &ParamP,
        const double &ParamQ,
        TWNet::TNodeI NI,
        int64 &NCnt,
        const bool &Verbose) {
    if (Verbose && NCnt % 100 == 0) {
        double PreprocessingProgress = (double) NCnt / (double) (InNet->GetNodes());
        PrintProgress("Preprocessing progress: ", PreprocessingProgress);
    }
    //for node t
    THash<TInt, TBool> NbrH;                                    //Neighbors of t
    for (int64 i = 0; i < NI.GetOutDeg(); i++) {
        NbrH.AddKey(NI.GetNbrNId(i));
    }
    for (int64 i = 0; i < NI.GetOutDeg(); i++) {
        TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));      //for each node v
        double Psum = 0;
        TFltV PTable;                              //Probability distribution table
        for (int64 j = 0; j < CurrI.GetOutDeg(); j++) {           //for each node x
            int64 FId = CurrI.GetNbrNId(j);
            TFlt Weight;
            if (!(InNet->GetEDat(CurrI.GetId(), FId, Weight))) { continue; }
            if (FId == NI.GetId()) {
                PTable.Add(Weight / ParamP);
                Psum += Weight / ParamP;
            } else if (NbrH.IsKey(FId)) {
                PTable.Add(Weight);
                Psum += Weight;
            } else {
                PTable.Add(Weight / ParamQ);
                Psum += Weight / ParamQ;
            }
        }
        //Normalizing table
        for (int64 j = 0; j < CurrI.GetOutDeg(); j++) {
            PTable[j] /= Psum;
        }
        GetNodeAlias(PTable, CurrI.GetDat().GetDat(NI.GetId()));
    }
    NCnt++;
}

/**
 * Preprocess transition probabilities for each path t -> v -> x.
 */
void PreprocessTransitionProbs(PWNet &InNet, const double &ParamP, const double &ParamQ, const bool &Verbose) {
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        InNet->SetNDat(NI.GetId(), TIntIntVFltVPrH());
    }
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        for (int64 i = 0; i < NI.GetOutDeg(); i++) {
            // Allocating space in advance to avoid issues with multi-threading
            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
            CurrI.GetDat().AddDat(NI.GetId(), TPair<TIntV, TFltV>(TIntV(CurrI.GetOutDeg()), TFltV(CurrI.GetOutDeg())));
        }
    }
    int64 NCnt = 0;
    TIntV NIds;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIds.Add(NI.GetId());
    }
#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < NIds.Len(); i++) {
        PreprocessNode(InNet, ParamP, ParamQ, InNet->GetNI(NIds[i]), NCnt, Verbose);
    }

    if (Verbose) {
        std::cout << std::endl << "Preprocessing transition probabilities: done!" << std::endl;
    }
}

void SimulateOneWalk(PWNet &InNet, int64 StartNId, const int &WalkLen, TRnd &Rnd, TIntV &WalkV) {
    WalkV.Add(StartNId);

    if (WalkLen == 1) {
        return;
    }

    if (InNet->GetNI(StartNId).GetOutDeg() == 0) {
        return;
    }

    WalkV.Add(InNet->GetNI(StartNId).GetNbrNId(Rnd.GetUniDevInt(InNet->GetNI(StartNId).GetOutDeg())));

    while (WalkV.Len() < WalkLen) {
        int64 Dst = WalkV.Last();
        int64 Src = WalkV.LastLast();

        if (InNet->GetNI(Dst).GetOutDeg() == 0) {
            return;
        }

        int64 Next = AliasDrawInt(InNet->GetNDat(Dst).GetDat(Src), Rnd);
        WalkV.Add(InNet->GetNI(Dst).GetNbrNId(Next));
    }
}

const TVVec<TInt, int64> SimulateWalks(
        PWNet &InNet,
        const int &WalkLen,
        const int &NumWalks,
        const double &ParamP,
        const double &ParamQ,
        const bool &Verbose
) {
    const int RandSeed = (int) time(NULL);
    return SimulateWalks(InNet, WalkLen, NumWalks, ParamP, ParamQ, Verbose, RandSeed);
}

const TVVec<TInt, int64> SimulateWalks(
        PWNet &InNet,
        const int &WalkLen,
        const int &NumWalks,
        const double &ParamP,
        const double &ParamQ,
        const bool &Verbose,
        const int &RandSeed
) {
    PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);

    TIntV NIdsV;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIdsV.Add(NI.GetId());
    }

    // Generate random walks
    int64 AllWalks = (int64) NumWalks * NIdsV.Len();
    TVVec<TInt, int64> WalksVV = TVVec<TInt, int64>(AllWalks, WalkLen);
    TRnd Rnd(RandSeed);
    int64 WalksDone = 0;
    for (int64 i = 0; i < NumWalks; i++) {
        NIdsV.Shuffle(Rnd);

#pragma omp parallel for schedule(dynamic)

        for (int64 j = 0; j < NIdsV.Len(); j++) {
            if (Verbose && WalksDone % 10000 == 0) {
                double WalkingProgress = (double) WalksDone / (double) AllWalks;
                PrintProgress("Walking progress: ", WalkingProgress);
            }

            TIntV WalkV;
            SimulateOneWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV);
            for (int64 k = 0; k < WalkV.Len(); k++) {
                WalksVV.PutXY(i * NIdsV.Len() + j, k, WalkV[k]);
            }
            WalksDone++;
        }
    }

    if (Verbose) {
        std::cout << std::endl << "Simulating random walks: done!" << std::endl;
    }

    return WalksVV;
}

int64 PredictMemoryRequirements(PWNet &InNet) {
    int64 MemNeeded = 0;

    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        for (int64 i = 0; i < NI.GetOutDeg(); i++) {
            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
            MemNeeded += CurrI.GetOutDeg() * (sizeof(TInt) + sizeof(TFlt));
        }
    }

    return MemNeeded;
}
