function [negLLE, fitData] = Make_negLLE(params, data)
    %% Flags
    % Conditional flags for low/high RPE
    % RPE_LOW     = 2;
    % RPE_HIGH    = 1;
    % 
    % % Conditional flags for low/high SPE
    % SPE_LOW     = 2;
    % SPE_HIGH    = 1;
    % 
    % % Conditional flags for stimulus/state reward contingency
    % REW_STIM    = 1; % Contingent to stimulus (data.resp1)
    % REW_STATE   = 2; % Contingent to State
    
    %% Transform Model Parameters
    % Store computational variables
    fitData = struct();
    fitData.choice = data.resp1;
    
    % Softmax betas for each RPE condition (which differ in reward magnitude)
    fitData.transParams(1)  = exp(params(1));
    smB                     = fitData.transParams(1);
    
    % Learning Rate
    fitData.transParams(2)  = 1./(1+ exp(-params(2)));
    rl_MF                   = fitData.transParams(2);
    
    fitData.transParams(3)  = 1./(1+ exp(-params(3)));
    rl_MB                   = fitData.transParams(3);
    
    % MF Eligibility Trace
    eD                      = 1; % Fixed in current model
    
    % SPE Learning Rate
    % fitData.transParams(4) = 1./(1+ exp(-params(4)));
    % aSPE1                  = fitData.transParams(4); % Not used in current model
    aSPE2                   = 0.5; % Fixed in current model based on previous analyses by Jeff
    
    % Influence of Reward Magnitude
    % Not used in current model
    % fitData.transParams(3) = params(3);
    % wMagStim               = fitData.transParams(3);
    %
    % fitData.transParams(4) = params(4);
    % wMagState              = fitData.transParams(4);
    
    % Win-Stay (WS) / Lose-Swtich (LS)
    fitData.transParams(4)  = params(4);
    wWS_MF                  = fitData.transParams(4);
    
    fitData.transParams(5)  = params(5);
    wLS_MF                  = fitData.transParams(5);
    
    fitData.transParams(6)  = params(6);
    wWS_MB                  = fitData.transParams(6);
    
    fitData.transParams(7)  = params(7);
    wLS_MB                  = fitData.transParams(7);
    
    % Anchor MF Weight
    fitData.transParams(8)  = params(8);
    wMF                     = fitData.transParams(8);
    
    rlMag                   = 1; % Fixed in current model based on previous analyses by Jeff
    
    fitData.transParams(9)  =  exp(params(9));
    wMag                    = fitData.transParams(9);
    
    % Conditional Effects on the Mixture Weight
    fitData.transParams(10) =  params(10);
    arb_mbRPE               = fitData.transParams(10);
    
    fitData.transParams(11) =  params(11);
    arb_mfRPE               = fitData.transParams(11);
    
    fitData.transParams(12) =  params(12);
    arb_SPE                 = fitData.transParams(12);
    
    %% Set default and track variables
    % Trial ID
    dataTID = data.trialID;
    
    %% Model-Free Values
    mfQ1                  = zeros(1,2); % Expected Value for Stimuli
    mfQ2                  = zeros(1,2); % Expected Value for Planets
    mfQ3                  = zeros(1,4); % Expected Value for Landing Pads
    fitData.mfQ1          = nan(size(dataTID,1), size(mfQ1,2));
    fitData.mfQ2          = nan(size(dataTID,1), size(mfQ2,2));
    fitData.mfQ3          = nan(size(dataTID,1), size(mfQ3,2));
    
    % Alternative way computing mfQ1
    % Equivalent to mfQ1
    mfQ1_alt              = zeros(1,2);
    fitData.mfQ1_alt      = nan(size(dataTID,1), size(mfQ1,2));
    
    % Values that was chosen during response/outcome1/outcome2
    fitData.chosen_mfQ1   = nan(size(dataTID,1), 1);
    fitData.chosen_mfQ2   = nan(size(dataTID,1), 1);
    fitData.chosen_mfQ3   = nan(size(dataTID,1), 1);
    
    % Trial-wise weighted values
    % weighted_mfQ1 = wMF_trial * mfQ1
    fitData.weighted_mfQ1 = nan(size(dataTID,1), size(mfQ1,2));
    
    %% Model-Based Values
    mbQ1                    = zeros(1,2); % EV for Stimuli
    mbQ2                    = zeros(1,2); % EV for Planets
    mbQ3                    = zeros(1,4); % EV for Landing Pads updated by binary RPE
    padValue                = zeros(1,4); % EV for Landing Pads updated by non-binary RPE; Equivalent to mb_padValue
    fitData.mbQ1            = nan(size(dataTID,1), size(mfQ1,2));
    fitData.mbQ2            = nan(size(dataTID,1), size(mfQ2,2));
    fitData.mbQ3            = nan(size(dataTID,1), size(mfQ3,2));
    fitData.padValue        = nan(size(dataTID,1), size(mfQ3,2));
    
    % Values that was chosen during response/outcome1/outcome2
    fitData.chosen_mbQ1     = nan(size(dataTID,1), 1);
    fitData.chosen_mbQ2     = nan(size(dataTID,1), 1);
    fitData.chosen_mbQ3     = nan(size(dataTID,1), 1);
    fitData.chosen_padValue = nan(size(dataTID,1), 1);
    
    % Trial-wise weighted values
    % weighted_mbQ1 = wMB_trial * mfQ1
    fitData.weighted_mbQ1   = nan(size(dataTID,1), size(mfQ1,2));
    
    %% Trial-wise Weights
    fitData.wMF_trial      = nan(size(dataTID,1), 1);
    fitData.wMB_trial      = nan(size(dataTID,1), 1);
    
    fitData.wMF_trial_CONT = nan(size(dataTID,1), 1);
    fitData.wMB_trial_CONT = nan(size(dataTID,1), 1);
    
    fitData.wMF_trial_RPE  = nan(size(dataTID,1), 1);
    fitData.wMB_trial_RPE  = nan(size(dataTID,1), 1);
    
    fitData.wMF_trial_SPE  = nan(size(dataTID,1), 1);
    fitData.wMB_trial_SPE  = nan(size(dataTID,1), 1);
    
    %% Utility
    % Utility: Q1-value plus trial-wise win-stay lose-switch value
    % Adds plus value to Q1 when win-stay, and negative value when lose-switch
    % mfUtility = mfQ1 + wWSLS_MF
    fitData.mfUtility               = nan(size(dataTID,1), size(mfQ1,2));
    fitData.mbUtility               = nan(size(dataTID,1), size(mfQ1,2));
    
    % weighted_mfUtility(tI,:) = wMF_trial * (mfQ1 + wWSLS_MF);
    fitData.weighted_mfUtility      = nan(size(dataTID,1), size(mfQ1,2));
    fitData.weighted_mbUtility      = nan(size(dataTID,1), size(mfQ1,2));
    
    % CONT_weighted_mfUtility(tI,:) = wMF_trial_CONT * (mfQ1 + wWSLS_MF);
    fitData.CONT_weighted_mfUtility = nan(size(dataTID,1), size(mfQ1,2));
    fitData.CONT_weighted_mbUtility = nan(size(dataTID,1), size(mfQ1,2));
    
    % RPE_weighted_mfUtility(tI,:) = wMF_trial_RPE * (mfQ1 + wWSLS_MF);
    fitData.RPE_weighted_mfUtility  = nan(size(dataTID,1), size(mfQ1,2));
    fitData.RPE_weighted_mbUtility  = nan(size(dataTID,1), size(mfQ1,2));
    
    % SPE_weighted_mfUtility(tI,:) = wMF_trial_SPE * (mfQ1 + wWSLS_MF);
    fitData.SPE_weighted_mfUtility  = nan(size(dataTID,1), size(mfQ1,2));
    fitData.SPE_weighted_mbUtility  = nan(size(dataTID,1), size(mfQ1,2));
    
    %% Transition Probabilities
    % Known Transition Probabilities for 1st Stage Choice
    pTrans_1         = [0.7,0.3,0.3,0.7];
    fitData.pTrans_1 = nan(size(dataTID,1), 4);
    
    % Learned Transitions for Each of the 2nd Stage States
    pTrans_2         = zeros(1,4) + 0.5;
    fitData.pTrans_2 = nan(size(dataTID,1), 4);
    
    %% Track Prediction Errors
    fitData.mfRPE1          = nan( size(dataTID,1), 1 );
    fitData.mfRPE2          = nan( size(dataTID,1), 1 );
    fitData.mfRPE3          = nan( size(dataTID,1), 1 );
    fitData.mfRPE4          = nan( size(dataTID,1), 1 );
    
    fitData.mbRPE1          = nan( size(dataTID,1), 1 );
    fitData.mbRPE2          = nan( size(dataTID,1), 1 );
    fitData.mbRPE3          = nan( size(dataTID,1), 1 );
    fitData.mbRPE4          = nan( size(dataTID,1), 1 );
    
    fitData.SPE1            = nan( size(dataTID,1), 1 );
    fitData.SPE2            = nan( size(dataTID,1), 1 );
    
    % Trial-wise weighted RPEs
    % weighted_mfRPE1 = wMF_trial * mfRPE1
    fitData.weighted_mfRPE1 = nan( size(dataTID,1), 1 );
    fitData.weighted_mfRPE2 = nan( size(dataTID,1), 1 );
    fitData.weighted_mfRPE3 = nan( size(dataTID,1), 1 );
    fitData.weighted_mfRPE4 = nan( size(dataTID,1), 1 );
    
    fitData.weighted_mbRPE1 = nan( size(dataTID,1), 1 );
    fitData.weighted_mbRPE2 = nan( size(dataTID,1), 1 );
    fitData.weighted_mbRPE3 = nan( size(dataTID,1), 1 );
    fitData.weighted_mbRPE4 = nan( size(dataTID,1), 1 );
    
    % fastRPE: trial-wise win-stay lose-switch value
    % Plus value when win-stay, and negative value when lose-switch
    % fastMFRPE(tI) =  wWS_MF when data.outcomeBin(tI) == 1
    % fastMFRPE(tI) = -wLS_MF when data.outcomeBin(tI) == 0
    fitData.fastMFRPE       = nan(size(dataTID,1),1);
    fitData.fastMBRPE       = nan(size(dataTID,1),1);
    
    %% Win-Stay Lose-Switch (WSLS) in MF/MB Models
    wWSLS_MF = zeros(1, 2);
    wWSLS_MB = zeros(1, 2);
    
    %% Summed Utility of Each Action
    % Linear Combination of Value Differences (Win-Stay Lose-Switch)
    % wMF_trial:                 Weight for MF per trial
    % wMB_trial = 1 - wMF_trial: Weight for MB per trial
    % qDiff = wMF_trial*(qMF_diff + MF_WSLS) + wMB_trial*(qMB_diff + MB_WSLS);
    fitData.qDiff       = nan( size(dataTID,1), 1 );
    
    % Linear Combination of Values
    % Q_Net =  wMF_trial * mfQ1 + wMB_trial * mbQ1;
    fitData.Q_Net       = nan( size(dataTID,1), 2 );
    
    % Linear Combination of Value Differences
    % Q_Net_Diff = wMF_trial * qMF_diff +  wMB_trial * qMB_diff;
    fitData.Q_Net_Diff  = nan( size(dataTID,1), 1 );
    
    % Linear Combination of Utility Differences (Win-Stay Lose-Switch)
    % actUtil = wMF_trial * (mfQ1 + wWSLS_MF) + wMB_trial * (mbQ1 + wWSLS_MB);
    fitData.actUtil     = nan( size(dataTID,1), 2 );
    
    %% Derive Options and Choices
    % Derive Choice Probability
    % fitData.pOption(tI,1): Probability for choosing 1
    % fitData.pOption(tI,2): Probability for choosing 2
    % fitData.pOption(tI,1) = 1 ./ (1 + exp(-smB * qDiff));
    % fitData.pOption(tI,2) = 1 - fitData.pOption(tI,1);
    fitData.pOption = nan(size(dataTID,1), 2);
    
    % Track Probability of Chosen Response
    fitData.pChoice = nan(size(dataTID,1), 1);
    
    %% Reward Magnitude
    magAct               = zeros(1, 2); % with Actions
    magPlanet            = zeros(1, 2); % with Planets
    magPad               = zeros(1, 4); % with Landing Pads
    
    % Reward Magnitude RPE
    fitData.magActRPE    = nan( size(dataTID,1), 1 );
    fitData.magPlanetRPE = nan( size(dataTID,1), 1 );
    fitData.magPadRPE    = nan( size(dataTID,1), 1 );
    
    %% Previous Reward Magnitude
    % Not used in current model
    % prevMagStim     = zeros(1,2);
    % prevMagState    = zeros(1,2);
    
    %% Flag Noting the ID of the Current Session
    runID = -1;
    
    %% Loop Through all Trials
    for tI = 1 : size(dataTID,1)
        %% Should Learned Values be Reset
        if data.runID(tI) ~= runID
            %% Update ID
            runID = data.runID(tI);
    
            %% Set Default Variables
            mfQ1(:)           = 0;
            mfQ2(:)           = 0;
            mfQ3(:)           = 0;
            mfQ1_alt(:)       = 0;
    
            mbQ1(:)           = 0;
            mbQ2(:)           = 0;
            mbQ3(:)           = 0;
    
            padValue(:)       = 0;
    
            pTrans_1          = [0.7,0.3,0.3,0.7];
            pTrans_2(:)       = 0.5;
    
            wWSLS_MF(:)       = 0;
            wWSLS_MB(:)       = 0;
    
            magAct(:)         = 0;
            magPlanet(:)      = 0;
            magPad(:)         = 0;
    
            % prevMagStim(:)  = 0; % Not used in current model
            % prevMagState(:) = 0; % Not used in current model
        end
    
        %% Was a Response Made
        if ~isnan(data.resp1(tI))
            %% Values from Previous Trial
            % MF Values
            fitData.mfQ1(tI,:)          = mfQ1;
            fitData.mfQ2(tI,:)          = mfQ2;
            fitData.mfQ3(tI,:)          = mfQ3;
            fitData.mfQ1_alt(tI,:)      = mfQ1_alt;
    
            fitData.chosen_mfQ1(tI)     = mfQ1(data.resp1(tI));
            fitData.chosen_mfQ2(tI)     = mfQ2(data.outcome1(tI));
            fitData.chosen_mfQ3(tI)     = mfQ3(data.outcome2(tI));
    
            % MB Values
            fitData.mbQ1(tI,:)          = mbQ1;
            fitData.mbQ2(tI,:)          = mbQ2;
            fitData.mbQ3(tI,:)          = mbQ3;
            fitData.padValue(tI,:)      = padValue;
    
            fitData.chosen_mbQ1(tI)     = mbQ1(data.resp1(tI));
            fitData.chosen_mbQ2(tI)     = mbQ2(data.outcome1(tI));
            fitData.chosen_mbQ3(tI)     = mbQ3(data.outcome2(tI));
            fitData.chosen_padValue(tI) = padValue(data.outcome2(tI));
    
            % Current State Transition Belief
            fitData.pTrans_1(tI,:)      = pTrans_1; % Fixed to [0.7,0.3,0.3,0.7] in current version
            fitData.pTrans_2(tI,:)      = pTrans_2; % Updated per trial
    
            %% MF Weights
            % Define Reward Contingency of Previous Trial
            % isPost_ContState == 1 and condContingent == 2 for MB
            % MB: Contigent to states (reward probabilities match those of landing pads)
            if data.isPost_ContState(tI) == 1
                wMF_mbRPE = -arb_mbRPE;
            else
                wMF_mbRPE = arb_mbRPE;
            end
    
            % Define Reward Reliability of Previous Trial
            % isPost_RewHigh == 1 and condReward == 2 for MB
            % MB: High reward; High MFRPE (When MFRPE is high, MF is not a good model)
            if data.isPost_RewHigh(tI) == 1
                wMF_mfRPE = -arb_mfRPE;
            else
                wMF_mfRPE = arb_mfRPE;
            end
    
            % Define State Reliability of Previous Trial
            % isPost_StateLow == 1 and condState == 2 for MB
            % MB: Reliable state representation; Low SPE
            % Modeled but not key params for performance
            if data.isPost_StateLow(tI) == 1
                wMF_SPE = -arb_SPE;
            else
                wMF_SPE = arb_SPE;
            end
    
            %% Trial-wise Weights
            % Combine Conditional Shift on Controller Mixture
            % Formula derived from previous analyses (Jeff)
            % wMF_trial: Weight for MF per trial
            wMF_trial      = 1./(1+exp(-1*(wMF + wMF_mbRPE + wMF_mfRPE + wMF_SPE)));
            wMB_trial      = 1 - wMF_trial;
    
            wMF_trial_CONT = 1./(1+exp(-1*(wMF + wMF_mbRPE)));
            wMB_trial_CONT = 1 - wMF_trial_CONT;
    
            wMF_trial_RPE  = 1./(1+exp(-1*(wMF + wMF_mfRPE)));
            wMB_trial_RPE  = 1 - wMF_trial_RPE;
    
            wMF_trial_SPE  = 1./(1+exp(-1*(wMF + wMF_SPE)));
            wMB_trial_SPE  = 1 - wMF_trial_SPE;
    
            fitData.wMF_trial(tI)      = wMF_trial;
            fitData.wMB_trial(tI)      = wMB_trial;
    
            fitData.wMF_trial_CONT(tI) = wMF_trial_CONT;
            fitData.wMB_trial_CONT(tI) = wMB_trial_CONT;
    
            fitData.wMF_trial_RPE(tI)  = wMF_trial_RPE;
            fitData.wMB_trial_RPE(tI)  = wMB_trial_RPE;
    
            fitData.wMF_trial_SPE(tI)  = wMF_trial_SPE;
            fitData.wMB_trial_SPE(tI)  = wMB_trial_SPE;
    
            %% Weighted Values/Utilities from Previous Trial
            fitData.weighted_mfQ1(tI,:)           = wMF_trial * mfQ1;
            fitData.weighted_mbQ1(tI,:)           = wMB_trial * mbQ1;
    
            fitData.mfUtility(tI,:)               = mfQ1 + wWSLS_MF;
            fitData.mbUtility(tI,:)               = mbQ1 + wWSLS_MB;
    
            fitData.weighted_mfUtility(tI,:)      = wMF_trial * (mfQ1 + wWSLS_MF);
            fitData.weighted_mbUtility(tI,:)      = wMB_trial * (mbQ1 + wWSLS_MB);
    
            fitData.CONT_weighted_mfUtility(tI,:) = wMF_trial_CONT * (mfQ1 + wWSLS_MF);
            fitData.CONT_weighted_mbUtility(tI,:) = (1-wMF_trial_CONT) * (mbQ1 + wWSLS_MB);
    
            fitData.RPE_weighted_mfUtility(tI,:)  = wMF_trial_RPE * (mfQ1 + wWSLS_MF);
            fitData.RPE_weighted_mbUtility(tI,:)  = (1-wMF_trial_RPE) * (mbQ1 + wWSLS_MB);
    
            fitData.SPE_weighted_mfUtility(tI,:)  = wMF_trial_SPE * (mfQ1 + wWSLS_MF);
            fitData.SPE_weighted_mbUtility(tI,:)  = (1-wMF_trial_SPE) * (mbQ1 + wWSLS_MB);
    
            %% Combine MF and MB Values
            % Action Differences
            qMF_diff       = mfQ1(1) - mfQ1(2);
            qMB_diff       = mbQ1(1) - mbQ1(2);
            % magStateDiff = wMagState * (prevMagState(1) - prevMagState(2));
            % magStimDiff  = wMagStim * (prevMagStim(1) - prevMagStim(2));
            MF_WSLS        = wWSLS_MF(1) - wWSLS_MF(2); % Either +/- wWS_MF/w_LS_MF win/lose
            MB_WSLS        = wWSLS_MB(1) - wWSLS_MB(2); % Either +/- wWS_MB/w_LS_MB common win/lose
    
            % Linear Combination of Value Differences (Win-Stay Lose-Switch)
            % wMF_trial:                 Weight for MF per trial
            % wMB_trial = 1 - wMF_trial: Weight for MB per trial
            qDiff                  = wMF_trial*(qMF_diff + MF_WSLS) + wMB_trial*(qMB_diff + MB_WSLS);
            fitData.qDiff(tI)      = qDiff;
    
            % Linear Combination of Values
            Q_Net                  = wMF_trial * mfQ1 + wMB_trial * mbQ1;
            fitData.Q_Net(tI,:)    = Q_Net;
    
            % Linear Combination of Value Differences
            Q_Net_Diff             = wMF_trial * qMF_diff +  wMB_trial * qMB_diff;
            fitData.Q_Net_Diff(tI) = Q_Net_Diff;
    
            % Linear Combination of Utility Differences (Win-Stay Lose-Switch)
            actUtil                = wMF_trial * (mfQ1 + wWSLS_MF) + wMB_trial * (mbQ1 + wWSLS_MB);
            fitData.actUtil(tI,:)  = actUtil;
    
            %% Derive Options and Choices
            % Derive Choice Probability
            % fitData.pOption(tI,1): Probability for choosing 1
            % fitData.pOption(tI,2): Probability for choosing 2
            fitData.pOption(tI,1) = 1 ./ (1 + exp(-smB * qDiff));
            fitData.pOption(tI,2) = 1 - fitData.pOption(tI,1);
    
            % Probability of Chosen Response
            fitData.pChoice(tI)   = fitData.pOption(tI, data.resp1(tI));
    
            %% MF RPEs
            fitData.mfRPE3(tI) = (data.outcomeMag(tI)/100) - mfQ3(data.outcome2(tI));
            fitData.mfRPE2(tI) = mfQ3(data.outcome2(tI))   - mfQ2(data.outcome1(tI));
            fitData.mfRPE1(tI) = mfQ2(data.outcome1(tI))   - mfQ1(data.resp1(tI));
            fitData.mfRPE4(tI) = (data.outcomeMag(tI)/100) - mfQ1(data.resp1(tI));
    
            fitData.weighted_mfRPE3(tI) = wMF_trial * ((data.outcomeMag(tI)/100) - mfQ3(data.outcome2(tI)));
            fitData.weighted_mfRPE2(tI) = wMF_trial * (mfQ3(data.outcome2(tI))   - mfQ2(data.outcome1(tI)));
            fitData.weighted_mfRPE1(tI) = wMF_trial * (mfQ2(data.outcome1(tI))   - mfQ1(data.resp1(tI)));
            fitData.weighted_mfRPE4(tI) = wMF_trial * ((data.outcomeMag(tI)/100) - mfQ1(data.resp1(tI)));
    
            %% Update MF Values According to the Chosen Option/Outcome RPE
            % Section (A): Equivalent to below section (B), only use either one
            mfQ1_alt(data.resp1(tI)) = mfQ1(data.resp1(tI))    +      rl_MF * fitData.mfRPE4(tI);
    
            % Section (B): Equivalent to above section (A), only use either one
            mfQ1(data.resp1(tI))     = mfQ1(data.resp1(tI))    +      rl_MF * fitData.mfRPE1(tI);
    
            % propagate the 2nd RPE through
            % mfQ2(data.outcome1(tI))  = mfQ2(data.outcome1(tI)) + eD * rl_MF * fitData.mfRPE2(tI); % eD should be removed
            mfQ2(data.outcome1(tI))  = mfQ2(data.outcome1(tI)) +      rl_MF * fitData.mfRPE2(tI);
            mfQ1(data.resp1(tI))     = mfQ1(data.resp1(tI))    + eD * rl_MF * fitData.mfRPE2(tI);
    
            % propagate the 3rd RPE through
            mfQ3(data.outcome2(tI))  = mfQ3(data.outcome2(tI)) +      rl_MF * fitData.mfRPE3(tI);
            mfQ2(data.outcome1(tI))  = mfQ2(data.outcome1(tI)) + eD * rl_MF * fitData.mfRPE3(tI);
            mfQ1(data.resp1(tI))     = mfQ1(data.resp1(tI))    + eD * rl_MF * fitData.mfRPE3(tI);
    
            %% MB RPEs
            % Binary and non-binary combined
            fitData.mbRPE3(tI) = data.outcomeBin(tI)         - mbQ3(data.outcome2(tI));
            fitData.mbRPE2(tI) = padValue(data.outcome2(tI)) - mbQ2(data.outcome1(tI));
            fitData.mbRPE1(tI) = mbQ2(data.outcome1(tI))     - mbQ1(data.resp1(tI));
            fitData.mbRPE4(tI) = data.outcomeBin(tI)         - mbQ1(data.resp1(tI));
    
            % Weighted RPEs
            fitData.weighted_mbRPE3(tI) = wMB_trial * ((data.outcomeMag(tI)/100) - mbQ3(data.outcome2(tI))); % Not used
            fitData.weighted_mbRPE2(tI) = wMB_trial * (mbQ3(data.outcome2(tI))   - mbQ2(data.outcome1(tI))); % Not used (mbQ3 should be padValue)
            fitData.weighted_mbRPE1(tI) = wMB_trial * (mbQ2(data.outcome1(tI))   - mbQ1(data.resp1(tI)));
            fitData.weighted_mbRPE4(tI) = wMB_trial * (data.outcomeBin(tI)       - mbQ1(data.resp1(tI)));
    
            %% Update MB Values According to the Chosen Option/Outcome RPE
            mbQ3(data.outcome2(tI)) = mbQ3(data.outcome2(tI)) + rl_MB * fitData.mbRPE3(tI);
    
            %% Stimulus-Mapped Win-Stay/Lose-Switch
            % MF: Independent to common/rare transition
            wWSLS_MF(:) = 0;
            if data.outcomeBin(tI) == 1
                wWSLS_MF(data.resp1(tI)) = wWS_MF;  % Win: Win-Stay
                fitData.fastMFRPE(tI)    = wWS_MF;
            else
                wWSLS_MF(data.resp1(tI)) = -wLS_MF; % Lose: Lose-Switch
                fitData.fastMFRPE(tI)    = -wLS_MF;
            end
    
            % MB: Dependent on common/rare transition
            wWSLS_MB(:) = 0;
            if (data.doRareTrans(tI) == 0 && data.outcomeBin(tI) == 1) || (data.doRareTrans(tI) == 1 && data.outcomeBin(tI) == 0)
                wWSLS_MB(data.resp1(tI)) = wWS_MB;  % (Common & Win)  | (Rare & Lose): Stay
                fitData.fastMBRPE(tI)    = wWS_MB; % New
            else
                wWSLS_MB(data.resp1(tI)) = -wLS_MB; % (Common & Lose) | (Rare & Win): Switch
                fitData.fastMBRPE(tI)    = -wLS_MB; % New
            end
    
            %% 1st-Level SPE
            if  data.resp1(tI) == 1
                fitData.SPE1(tI) = 1 - pTrans_1(data.outcome1(tI));
                % pTrans_1(data.outcome1(tI)) = pTrans_1(data.outcome1(tI)) + aSPE1 * fitData.SPE1(tI);
                % altStateIndex = 3-data.outcome1(tI);
            else
                fitData.SPE1(tI) = 1 - pTrans_1(data.outcome1(tI)+2);
                % pTrans_1(data.outcome1(tI)+2) = pTrans_1(data.outcome1(tI)+2) + aSPE1 * fitData.SPE1(tI);
                % altStateIndex = 5-data.outcome1(tI);
            end
    
            % pTrans_1(altStateIndex) = pTrans_1(altStateIndex) * (1-aSPE1);
    
            %% 2nd-Level SPE
            fitData.SPE2(tI)            = 1 - pTrans_2(data.outcome2(tI));
    
            % Update the transition taken
            pTrans_2(data.outcome2(tI)) = pTrans_2(data.outcome2(tI)) + aSPE2 * fitData.SPE2(tI);
    
            % Update the transition not taken
            % Transition probability taken + not taken = 1
            if data.outcome2(tI) == 1 || data.outcome2(tI) == 3
                altStateIndex = data.outcome2(tI) + 1;
            else
                altStateIndex = data.outcome2(tI) - 1;
            end
    
            pTrans_2(altStateIndex)     = pTrans_2(altStateIndex) * (1-aSPE2);
    
            %% Magnitude Prediction Errors
            % Why updated only during reward:
            % Only magPad is used to track reward values
            if data.outcomeMag(tI) > 0
                fitData.magActRPE(tI)        = (data.outcomeMag(tI)/100)    - magAct(data.resp1(tI));
                magAct(data.resp1(tI))       = magAct(data.resp1(tI))       + rlMag * (fitData.magActRPE(tI));
    
                fitData.magPlanetRPE(tI)     = (data.outcomeMag(tI)/100)    - magPlanet(data.outcome1(tI));
                magPlanet(data.outcome1(tI)) = magPlanet(data.outcome1(tI)) + rlMag * (fitData.magPlanetRPE(tI));
    
                fitData.magPadRPE(tI)        = (data.outcomeMag(tI)/100)    - magPad(data.outcome2(tI));
                magPad(data.outcome2(tI))    = magPad(data.outcome2(tI))    + rlMag * (fitData.magPadRPE(tI));
            end
    
            %% Update MB Values According to State Transisions and Magnitude RPEs
            % Expected Value for Landing Pads
            % mbQ3 only updated according to binary RPE
            % padValue consists of both binary and non-binary RPEs
            padValue = mbQ3 + (wMag * magPad);
    
            % Expected Value for Planets
            mbQ2 = [pTrans_2([1,2]) * padValue([1,2])', pTrans_2([3,4]) * padValue([3,4])'];
    
            % Expected Value for Stimuli
            mbQ1 = [pTrans_1([1,2]) * mbQ2', pTrans_1([3,4]) * mbQ2'];
    
            %% Track Previous Reward Magniude: Not used
            % prevMagStim(:)                     = 0;
            % prevMagState(:)                    = 0;
            % prevMagStim(data.resp1(tI))        = data.outcomeMag(tI)/100;
            % prevMagState(data.outcome1(tI))    = data.outcomeMag(tI)/100;
    
        end % Was a Response Made
    end % Loop Through all Trials
    
    %% Estimate Null Model
    % Determine Valid Trials for the Model-Fitting
    isValidTrial = ~isnan(data.resp1); % & data.TOI==1;
    n            = sum(isValidTrial);
    k            = size(params,2); % Number of free parameters
    
    % Adjust 0-Probability Trials
    fitData.pChoice(fitData.pChoice < eps | isnan(fitData.pChoice) | isinf(fitData.pChoice)) = eps;
    
    % Compute Model negLLE (Negative Log-Likelihood Estimate)
    % negLLE = - sum( log( probabilty ) )
    % Null Probability = 0.5
    null_negLLE         = - sum(isValidTrial) * log(0.5);
    negLLE              = - sum( log( fitData.pChoice(isValidTrial) ) );
    fitData.null_negLLE = null_negLLE;
    fitData.negLLE      = negLLE;
    
    % Compute Model Estimates
    fitData.pseudoR = 1 - (fitData.negLLE/fitData.null_negLLE);
    fitData.AIC     = 2*fitData.negLLE + 2*k;
    fitData.BIC     = 2*fitData.negLLE + k*log(n);
    
    % Track Model Parameters
    fitData.wMF       = wMF;
    fitData.arb_mbRPE = arb_mbRPE;
    fitData.arb_mfRPE = arb_mfRPE;
    fitData.arb_SPE   = arb_SPE;

end % Function Ends

