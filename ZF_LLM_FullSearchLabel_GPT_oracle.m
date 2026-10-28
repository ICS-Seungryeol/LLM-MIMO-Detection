clear; clc; close all;

%% ============================================================
%  4x4 MIMO 4-QAM  ZF x SIC / OSIC / LLM-based BER 비교
%  모델: gpt-4o-mini (OpenAI)
%      - ZF 필터 기반 (pinv(H))
%      - ZF-OSIC ordering: 1/‖W(j,:)‖² 기반 effective SNR
%      - Few-shot 라벨: 4! Full-search (ZF 기준, probe 방식)
%      - Oracle FULL: eval 심볼로 24개 전수조사 → 정의상 FULL ≤ OSIC 보장
%      - ExamplePool 저장/재사용
%      - 비교: ZF-SIC / ZF-OSIC / ZF-FULL(oracle) / ZF-LLM
%% ============================================================

%% ── 파라미터 ─────────────────────────────────────────────
Tx = 4;  Rx = 4;  M = 4;

EbN0_dB             = 0:5:20;
NumberIteration     = 1000;
NumSymbolPerChannel = 100;
Nshot               = 10;
ExamplePoolSize     = 500;
LabelNumSymbol      = 100;

% OpenAI API key 환경변수로 설정
setenv('OPENAI_API_KEY','YOUR_KEY')
apiKey = getenv('OPENAI_API_KEY');
if strlength(string(apiKey)) == 0
    error('OPENAI_API_KEY 환경변수가 비어 있습니다. setenv(''OPENAI_API_KEY'',''YOUR_KEY'')로 설정하세요.');
end

modelName  = 'gpt-4o-mini';
maxRetry   = 10;
retryDelay = 120;
%% ──────────────────────────────────────────────────────────

%% ── 파생 변수 ────────────────────────────────────────────
nu        = log2(M);
LengthBit = nu * Tx;
EbN0      = db2pow(EbN0_dB);
EsN0      = EbN0 * log2(M);
NormFac   = sqrt(2/3*(M-1)*Tx);
alphabet  = qammod(0:M-1, M) / NormFac;

%% ── 누적 변수 ────────────────────────────────────────────
EC_ZF_SIC  = zeros(1, length(EbN0_dB));
EC_ZF_OSIC = zeros(1, length(EbN0_dB));
EC_ZF_FULL = zeros(1, length(EbN0_dB));
EC_ZF_LLM  = zeros(1, length(EbN0_dB));

DoneIter    = zeros(1, length(EbN0_dB));
FallbackCnt = zeros(1, length(EbN0_dB));

ZF_MatchCount      = zeros(1, length(EbN0_dB));   % LLM vs OSIC full match
ZF_PartialMatch    = zeros(1, length(EbN0_dB));    % LLM vs OSIC 1st match
ZF_FullRefMatch    = zeros(1, length(EbN0_dB));    % LLM vs FULL full match
ZF_FullRef1stMatch = zeros(1, length(EbN0_dB));    % LLM vs FULL 1st match

BER_conv_ZF_SIC  = nan(length(EbN0_dB), NumberIteration);
BER_conv_ZF_LLM  = nan(length(EbN0_dB), NumberIteration);
BER_conv_ZF_OSIC = nan(length(EbN0_dB), NumberIteration);
BER_conv_ZF_FULL = nan(length(EbN0_dB), NumberIteration);

%% ── Few-shot 예시 후보 pool 생성 ─────────────────────────
poolFile = sprintf('ExamplePool_ZF_GPT_L%d.mat', LabelNumSymbol);

if exist(poolFile, 'file')
    fprintf('기존 pool 로드 중: %s\n', poolFile);
    loaded = load(poolFile, 'ExamplePool', 'LabelNumSymbol');
    if loaded.LabelNumSymbol == LabelNumSymbol && ...
       numel(loaded.ExamplePool) == length(EbN0_dB)
        ExamplePool = loaded.ExamplePool;
        fprintf('로드 완료\n\n');
    else
        fprintf('설정 불일치 → pool 재생성\n');
        ExamplePool = repmat(struct('examples', strings(1, ExamplePoolSize), ...
                                    'features', zeros(ExamplePoolSize, 16)), ...
                             1, length(EbN0_dB));
        for i = 1:length(EbN0_dB)
            ExamplePool(i) = buildFewShotPool_ZF(Tx, Rx, ExamplePoolSize, M, LabelNumSymbol, alphabet, NormFac, EsN0(i));
        end
        save(poolFile, 'ExamplePool', 'LabelNumSymbol');
        fprintf('완료 및 저장\n\n');
    end
else
    fprintf('Few-shot 후보 pool 생성 중...\n');
    ExamplePool = repmat(struct('examples', strings(1, ExamplePoolSize), ...
                                'features', zeros(ExamplePoolSize, 16)), ...
                         1, length(EbN0_dB));
    for i = 1:length(EbN0_dB)
        ExamplePool(i) = buildFewShotPool_ZF(Tx, Rx, ExamplePoolSize, M, LabelNumSymbol, alphabet, NormFac, EsN0(i));
    end
    save(poolFile, 'ExamplePool', 'LabelNumSymbol');
    fprintf('완료 및 저장: %s\n\n', poolFile);
end

startTime = tic;

%% ── 메인 루프 ────────────────────────────────────────────
try
for idxSNR = 1:length(EbN0_dB)
    cur_EsN0     = EsN0(idxSNR);
    snrStartTime = tic;
    fprintf('\n========== Eb/N0 = %d dB ==========\n', EbN0_dB(idxSNR));

    allOrders_24 = perms(1:Tx);   % 24×4, SNR 루프마다 한 번만 생성

    for iTotal = 1:NumberIteration

        H = (randn(Rx,Tx) + 1j*randn(Rx,Tx)) ./ sqrt(2);

        fixedOrder    = 1:Tx;
        zfOSICOrder   = computeZF_OSICOrder(H, Tx);
        err_allOrders = zeros(1, size(allOrders_24,1));

        % 현재 H와 특징이 가까운 few-shot 예시 선택
        Examples = selectFewShot_ZF(H, ExamplePool(idxSNR), Nshot);

        [zfLLM, ~, isFB] = getLLMOrder_ZF( ...
            H, EbN0_dB(idxSNR), Examples, ...
            apiKey, modelName, maxRetry, retryDelay);

        if isFB
            FallbackCnt(idxSNR) = FallbackCnt(idxSNR) + 1;
        end

        if isequal(zfLLM, zfOSICOrder)
            ZF_MatchCount(idxSNR) = ZF_MatchCount(idxSNR) + 1;
        end
        if zfLLM(1) == zfOSICOrder(1)
            ZF_PartialMatch(idxSNR) = ZF_PartialMatch(idxSNR) + 1;
        end

        fprintf('  ZF-OSIC=%s\n', mat2str(zfOSICOrder));
        fprintf('  ZF-LLM =%s  FB=%d\n', mat2str(zfLLM), isFB);

        for idxFrame = 1:NumSymbolPerChannel
            BitSeq = randi([0 1], 1, LengthBit);

            SigSeq = zeros(1, Tx);
            for k = 1:Tx
                SigSeq(k) = bi2de(BitSeq((k-1)*nu+1:k*nu), 'left-msb');
            end

            Symbol = qammod(SigSeq.', M) / NormFac;
            Noise  = (randn(Rx,1) + 1j*randn(Rx,1)) / sqrt(2);
            y      = H * Symbol + Noise * sqrt(1/cur_EsN0);

            EC_ZF_SIC(idxSNR)  = EC_ZF_SIC(idxSNR)  + ZF_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,fixedOrder);
            EC_ZF_OSIC(idxSNR) = EC_ZF_OSIC(idxSNR) + ZF_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,zfOSICOrder);
            EC_ZF_LLM(idxSNR)  = EC_ZF_LLM(idxSNR)  + ZF_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,zfLLM);

            % Oracle FULL: 같은 symbol/noise로 24개 전수조사 → 누적
            for oi = 1:size(allOrders_24,1)
                err_allOrders(oi) = err_allOrders(oi) + ...
                    ZF_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,allOrders_24(oi,:));
            end
        end

        % Oracle FULL: 최소 누적 error → 정의상 FULL ≤ OSIC 보장
        % 주의: 고SNR에서 동률 ordering이 많으면 zfFullOrder가 랜덤에 가까워
        %       LLM vs FULL 일치율이 왜곡될 수 있음
        [minFullErr, bestOI] = min(err_allOrders);
        EC_ZF_FULL(idxSNR) = EC_ZF_FULL(idxSNR) + minFullErr;
        zfFullOrder = allOrders_24(bestOI,:);

        if isequal(zfLLM, zfFullOrder)
            ZF_FullRefMatch(idxSNR) = ZF_FullRefMatch(idxSNR) + 1;
        end
        if zfLLM(1) == zfFullOrder(1)
            ZF_FullRef1stMatch(idxSNR) = ZF_FullRef1stMatch(idxSNR) + 1;
        end
        fprintf('  ZF-FULL=%s  (oracle, same eval symbols)\n', mat2str(zfFullOrder));

        DoneIter(idxSNR) = DoneIter(idxSNR) + 1;

        elapsed   = toc(startTime);
        doneTotal = sum(DoneIter);
        totalCalls= length(EbN0_dB) * NumberIteration;
        remaining = (elapsed / doneTotal) * (totalCalls - doneTotal);

        fprintf('  iter %d/%d 완료 | SNR경과 %.1f분 | 전체경과 %.1f분 | 남은시간 약 %.1f분\n', ...
            DoneIter(idxSNR), NumberIteration, toc(snrStartTime)/60, elapsed/60, remaining/60);

        tb_now = LengthBit * NumSymbolPerChannel * DoneIter(idxSNR);
        BER_conv_ZF_SIC(idxSNR,  DoneIter(idxSNR)) = EC_ZF_SIC(idxSNR)  / tb_now;
        BER_conv_ZF_LLM(idxSNR,  DoneIter(idxSNR)) = EC_ZF_LLM(idxSNR)  / tb_now;
        BER_conv_ZF_OSIC(idxSNR, DoneIter(idxSNR)) = EC_ZF_OSIC(idxSNR) / tb_now;
        BER_conv_ZF_FULL(idxSNR, DoneIter(idxSNR)) = EC_ZF_FULL(idxSNR) / tb_now;

        save('progress_ZF_FullSearchLabel_GPT_oracle.mat', ...
            'EC_ZF_SIC','EC_ZF_OSIC','EC_ZF_FULL','EC_ZF_LLM', ...
            'DoneIter','FallbackCnt', ...
            'ZF_MatchCount','ZF_PartialMatch','ZF_FullRefMatch','ZF_FullRef1stMatch', ...
            'BER_conv_ZF_SIC','BER_conv_ZF_LLM','BER_conv_ZF_OSIC','BER_conv_ZF_FULL', ...
            'EbN0_dB','LengthBit','NumSymbolPerChannel','Nshot','ExamplePoolSize','LabelNumSymbol');
    end
end
catch ME
    fprintf('\n중단: %s\n', ME.message);
end

fprintf('\n총 소요 시간: %.1f분 (%.1f시간)\n', toc(startTime)/60, toc(startTime)/3600);

%% ── BER 계산 ─────────────────────────────────────────────
TotalBits = LengthBit * NumSymbolPerChannel * DoneIter;
valid     = DoneIter > 0;

BER_ZF_SIC  = nan(size(EbN0_dB));
BER_ZF_OSIC = nan(size(EbN0_dB));
BER_ZF_FULL = nan(size(EbN0_dB));
BER_ZF_LLM  = nan(size(EbN0_dB));

BER_ZF_SIC(valid)  = EC_ZF_SIC(valid)  ./ TotalBits(valid);
BER_ZF_OSIC(valid) = EC_ZF_OSIC(valid) ./ TotalBits(valid);
BER_ZF_FULL(valid) = EC_ZF_FULL(valid) ./ TotalBits(valid);
BER_ZF_LLM(valid)  = EC_ZF_LLM(valid)  ./ TotalBits(valid);

%% ── 순서 일치율 계산 ─────────────────────────────────────
ZF_FullMatchRate   = nan(size(EbN0_dB));
ZF_FirstMatchRate  = nan(size(EbN0_dB));
ZF_FullRefMatchRate= nan(size(EbN0_dB));
ZF_FullRefFirstRate= nan(size(EbN0_dB));

ZF_FullMatchRate(valid)    = ZF_MatchCount(valid)      ./ DoneIter(valid) * 100;
ZF_FirstMatchRate(valid)   = ZF_PartialMatch(valid)    ./ DoneIter(valid) * 100;
ZF_FullRefMatchRate(valid) = ZF_FullRefMatch(valid)    ./ DoneIter(valid) * 100;
ZF_FullRefFirstRate(valid) = ZF_FullRef1stMatch(valid) ./ DoneIter(valid) * 100;

%% ── BER Gap 계산 (보간) ──────────────────────────────────
targetBER = 1e-2;
ZF_LLM_Gain_vs_SIC_dB  = calcBERGap(EbN0_dB, BER_ZF_SIC,  BER_ZF_LLM,  targetBER);
ZF_LLM_Loss_vs_OSIC_dB = calcBERGap(EbN0_dB, BER_ZF_LLM,  BER_ZF_OSIC, targetBER);
ZF_LLM_Loss_vs_FULL_dB = calcBERGap(EbN0_dB, BER_ZF_LLM,  BER_ZF_FULL, targetBER);

%% ── 통계 출력 ────────────────────────────────────────────
fprintf('\n===== Fallback 통계 =====\n');
fprintf('%-8s  %-12s  %-10s  %-10s\n', 'Eb/N0', 'Fallback', 'DoneIter', 'Rate(%)');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d  %-12d  %-10d  %.1f%%\n', ...
            EbN0_dB(i), FallbackCnt(i), DoneIter(i), 100*FallbackCnt(i)/DoneIter(i));
    end
end

fprintf('\n===== LLM vs OSIC 순서 일치율 =====\n');
fprintf('%-8s  %-16s  %-16s\n', 'Eb/N0','OSIC_Full(%)','OSIC_1st(%)');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d  %-16.1f  %-16.1f\n', ...
            EbN0_dB(i), ZF_FullMatchRate(i), ZF_FirstMatchRate(i));
    end
end

fprintf('\n===== LLM vs FULL-search 순서 일치율 =====\n');
fprintf('%-8s  %-16s  %-16s\n', 'Eb/N0','FULL_Full(%)','FULL_1st(%)');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d  %-16.1f  %-16.1f\n', ...
            EbN0_dB(i), ZF_FullRefMatchRate(i), ZF_FullRefFirstRate(i));
    end
end

fprintf('\n===== BER Gap (targetBER=%.0e) =====\n', targetBER);
fprintf('ZF: LLM Gain vs SIC  = %.3f dB\n', ZF_LLM_Gain_vs_SIC_dB);
fprintf('ZF: LLM Loss vs OSIC = %.3f dB\n', ZF_LLM_Loss_vs_OSIC_dB);
fprintf('ZF: LLM Loss vs FULL = %.3f dB\n', ZF_LLM_Loss_vs_FULL_dB);

fprintf('\n===== 정확한 BER 값 =====\n');
fprintf('%-8s %-14s %-14s %-14s %-14s\n', 'Eb/N0','ZF-SIC','ZF-OSIC','ZF-FULL (oracle)','ZF-LLM');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d %-14.6e %-14.6e %-14.6e %-14.6e\n', ...
            EbN0_dB(i), BER_ZF_SIC(i), BER_ZF_OSIC(i), BER_ZF_FULL(i), BER_ZF_LLM(i));
    end
end

%% ── 저장 ─────────────────────────────────────────────────
save('final_ZF_FullSearchLabel_GPT_oracle.mat', 'EbN0_dB','DoneIter','FallbackCnt', ...
    'BER_ZF_SIC','BER_ZF_OSIC','BER_ZF_FULL','BER_ZF_LLM', ...
    'ZF_FullMatchRate','ZF_FirstMatchRate', ...
    'ZF_FullRefMatchRate','ZF_FullRefFirstRate', ...
    'ZF_LLM_Gain_vs_SIC_dB','ZF_LLM_Loss_vs_OSIC_dB','ZF_LLM_Loss_vs_FULL_dB', ...
    'Nshot','ExamplePoolSize','LabelNumSymbol');

%% ── 그래프 1: BER 곡선 ──────────────────────────────────
floorBER = 1e-5;
toPlot   = @(b) max(b, floorBER);

figure;
semilogy(EbN0_dB, toPlot(BER_ZF_SIC),  '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
semilogy(EbN0_dB, toPlot(BER_ZF_OSIC), '-s', 'LineWidth',1.5, 'MarkerSize',7);
semilogy(EbN0_dB, toPlot(BER_ZF_FULL), '-d', 'LineWidth',1.5, 'MarkerSize',7);
semilogy(EbN0_dB, toPlot(BER_ZF_LLM),  '-^', 'LineWidth',2.0, 'MarkerSize',8);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('BER');
title('ZF-only: Oracle Full-search + LLM Ordering (GPT-4o-mini)');
legend('ZF-SIC','ZF-OSIC','ZF-FULL (oracle)','ZF-LLM','Location','southwest');
ylim([floorBER 1]);

%% ── 그래프 2: 순서 일치율 ───────────────────────────────
figure;
plot(EbN0_dB, ZF_FullMatchRate,   '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
plot(EbN0_dB, ZF_FirstMatchRate,  '-s', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, ZF_FullRefMatchRate,'-^', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, ZF_FullRefFirstRate,'-d', 'LineWidth',1.5, 'MarkerSize',7);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('Match Rate (%)');
title('ZF LLM 순서 일치율 (GPT-4o-mini)');
legend('vs OSIC 완전일치','vs OSIC 1번째일치','vs FULL 완전일치','vs FULL 1번째일치','Location','best');
ylim([0 100]);

%% ── 그래프 3: BER Gap ───────────────────────────────────
ZF_gap_vs_sic  = nan(size(EbN0_dB));
ZF_gap_vs_osic = nan(size(EbN0_dB));
ZF_gap_vs_full = nan(size(EbN0_dB));

for i = 1:length(EbN0_dB)
    if valid(i) && BER_ZF_LLM(i)>0 && BER_ZF_SIC(i)>0 && BER_ZF_OSIC(i)>0 && BER_ZF_FULL(i)>0
        ZF_gap_vs_sic(i)  = 10*log10(BER_ZF_SIC(i)  / BER_ZF_LLM(i));
        ZF_gap_vs_osic(i) = 10*log10(BER_ZF_LLM(i)  / BER_ZF_OSIC(i));
        ZF_gap_vs_full(i) = 10*log10(BER_ZF_LLM(i)  / BER_ZF_FULL(i));
    end
end

figure;
plot(EbN0_dB, ZF_gap_vs_sic,  '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
plot(EbN0_dB, ZF_gap_vs_osic, '-s', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, ZF_gap_vs_full, '-^', 'LineWidth',1.5, 'MarkerSize',7);
yline(0, 'k--', 'LineWidth',1.0);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('BER Ratio Gap (dB)');
title('ZF BER Gap: vs SIC 양수=개선, vs OSIC/FULL 양수=손실 (GPT-4o-mini)');
legend('ZF: LLM Gain vs SIC','ZF: LLM Loss vs OSIC','ZF: LLM Loss vs FULL','Location','best');

%% ── 그래프 4: Fallback 비율 ─────────────────────────────
FallbackRate = nan(size(EbN0_dB));
FallbackRate(valid) = FallbackCnt(valid) ./ DoneIter(valid) * 100;

figure;
bar(EbN0_dB, FallbackRate, 0.5);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('Fallback 비율 (%)');
title('SNR별 LLM Fallback 비율 (GPT-4o-mini)');
ylim([0 100]);
xticks(EbN0_dB);

%% ── 그래프 5: 누적 BER 수렴 (SNR=10dB) ─────────────────
snr_idx = find(EbN0_dB == 10, 1);
if ~isempty(snr_idx) && DoneIter(snr_idx) > 0
    n_done    = DoneIter(snr_idx);
    iter_axis = 1:n_done;

    figure;
    semilogy(iter_axis, BER_conv_ZF_SIC(snr_idx,1:n_done),  '-o', 'LineWidth',1.5, 'MarkerSize',4); hold on;
    semilogy(iter_axis, BER_conv_ZF_OSIC(snr_idx,1:n_done), '-s', 'LineWidth',1.5, 'MarkerSize',4);
    semilogy(iter_axis, BER_conv_ZF_FULL(snr_idx,1:n_done), '-d', 'LineWidth',1.5, 'MarkerSize',4);
    semilogy(iter_axis, BER_conv_ZF_LLM(snr_idx,1:n_done),  '-^', 'LineWidth',2.0, 'MarkerSize',5);
    grid on;
    xlabel('Iteration 수');
    ylabel('누적 BER');
    title('ZF BER 수렴 과정 (Eb/N0 = 10 dB, GPT-4o-mini)');
    legend('ZF-SIC','ZF-OSIC','ZF-FULL (oracle)','ZF-LLM','Location','best');
end

%% ============================================================
%  함수 정의
%% ============================================================

%% ZF-SIC 검출 함수
function bec = ZF_SIC_detect(H, y, SigSeq, BitSeq, M, NormFac, alphabet, order)
    Tx     = size(H,2);
    nu     = log2(M);
    H_rem  = H;
    y_rem  = y;
    remIdx = 1:Tx;
    detSig = zeros(1,Tx);

    for stage = 1:Tx
        target = order(stage);
        rel    = find(remIdx == target);

        % ZF 필터: pinv
        W = NormFac * sqrt(Tx) * pinv(H_rem);
        z = W(rel,:) * y_rem;

        detSig(target) = qamdemod(z, M);
        remod = alphabet(detSig(target)+1);
        y_rem = y_rem - H_rem(:,rel) * remod;

        H_rem(:,rel) = [];
        remIdx(rel)  = [];
    end

    nu     = log2(M);
    detBits = zeros(1, Tx*nu);
    for k = 1:Tx
        detBits((k-1)*nu+1:k*nu) = de2bi(detSig(k), nu, 'left-msb');
    end
    bec = sum(BitSeq ~= detBits);
end

%% ZF-OSIC 순서 계산 (effective SNR: 1/‖W(j,:)‖²)
function order = computeZF_OSICOrder(H, Tx)
    order  = zeros(1,Tx);
    H_rem  = H;
    remIdx = 1:Tx;

    for stage = 1:Tx
        W      = pinv(H_rem);
        metric = zeros(1, size(H_rem,2));
        for j = 1:size(H_rem,2)
            metric(j) = 1 / norm(W(j,:))^2;  % effective SNR (ZF)
        end
        [~, rel]     = max(metric);
        order(stage) = remIdx(rel);

        H_rem(:,rel) = [];
        remIdx(rel)  = [];
    end
end

%% ZF Full-Search 최적 순서 계산
function [bestOrder, errByOrder] = computeZF_FullSearchOrder(H, Tx, M, numProbeSym, alphabet, NormFac, EsN0, tieBreakOrder)
    nu        = log2(M);
    LengthBit = nu * Tx;
    allOrders = perms(1:Tx);
    nOrd      = size(allOrders,1);
    errByOrder= zeros(nOrd,1);

    for n = 1:numProbeSym
        BitSeq = randi([0 1], 1, LengthBit);
        SigSeq = zeros(1, Tx);
        for k = 1:Tx
            SigSeq(k) = bi2de(BitSeq((k-1)*nu+1:k*nu), 'left-msb');
        end
        Symbol = qammod(SigSeq.', M) / NormFac;
        Noise  = (randn(size(H,1),1) + 1j*randn(size(H,1),1)) / sqrt(2);
        y      = H * Symbol + Noise * sqrt(1/EsN0);

        for oi = 1:nOrd
            errByOrder(oi) = errByOrder(oi) + ZF_SIC_detect( ...
                H, y, SigSeq, BitSeq, M, NormFac, alphabet, allOrders(oi,:));
        end
    end

    minErr   = min(errByOrder);
    candIdx  = find(errByOrder == minErr);
    bestIdx  = candIdx(1);

    if nargin >= 8 && ~isempty(tieBreakOrder)
        for ii = 1:numel(candIdx)
            if isequal(allOrders(candIdx(ii),:), tieBreakOrder)
                bestIdx = candIdx(ii);
                break;
            end
        end
    end
    bestOrder = allOrders(bestIdx,:);
end

%% LLM 순서 예측 (GPT-4o-mini)
function [zfOrder, answer, isFallback] = getLLMOrder_ZF( ...
        H, snr_dB, Examples, apiKey, modelName, maxRetry, retryDelay)

    Tx = size(H,2);

    % Fallback: SepScore 기반
    [~, ~, ~, ~, sepScore] = channelFeatureSummary(H);
    [~, fb] = sort(sepScore, 'descend');
    zfOrder    = fb;
    answer     = "";
    isFallback = true;

    % ── System Prompt ──────────────────────────────────────
    SystemPrompt = ...
        "You are an expert in MIMO signal processing using Zero-Forcing (ZF) equalization. " + ...
        "Your task is to determine the optimal stream detection order for ZF-SIC " + ...
        "(Zero-Forcing Successive Interference Cancellation)." + newline + newline + ...
        "ZF perspective:" + newline + ...
        "ZF equalization is sensitive to noise amplification when channel columns are weak or highly correlated. " + ...
        "A reliable stream has strong channel energy AND good separation from remaining columns." + newline + newline + ...
        "Input features explained:" + newline + ...
        "- SNR_dB: Signal-to-noise ratio in dB." + newline + ...
        "- ColNorm: Euclidean norm of each column of H (stream signal strength)." + newline + ...
        "- NormCorr: Normalized absolute correlation between channel columns (4x4 matrix). " + ...
        "  Lower off-diagonal value = better stream separability under ZF." + newline + ...
        "- AvgCorr: Average normalized correlation of each stream with all other streams." + newline + ...
        "- MaxCorr: Maximum normalized correlation of each stream with any other stream." + newline + ...
        "- SepScore: Simple separability indicator = ColNorm / (AvgCorr + 0.1). Larger = more separable." + newline + ...
        "- RealH, ImagH: Real and imaginary parts of channel matrix H (4x4 MATLAB format)." + newline + newline + ...
        "Detection order rules:" + newline + ...
        "1. The first detected stream is most important due to SIC error propagation." + newline + ...
        "2. Detect the most reliable stream first (strong ColNorm AND low AvgCorr/MaxCorr)." + newline + ...
        "3. Do not rely only on ColNorm. High correlation causes noise amplification under ZF." + newline + ...
        "4. Use SepScore as a combined indicator of strength and separability." + newline + ...
        "5. At low SNR, prioritize ColNorm (noise robustness). " + ...
        "At high SNR, prioritize NormCorr and AvgCorr (interference separation)." + newline + ...
        "6. After choosing one stream, conceptually remove it and repeat for remaining streams." + newline + ...
        "7. Output ONLY the detection order as a JSON object.";

    % ── Input features ─────────────────────────────────────
    [colNorms, normCorr, avgCorr, maxCorr, sepScore] = channelFeatureSummary(H);

    inp = "{""SNR_dB"":" + string(snr_dB) + "," + ...
          """ColNorm"":"  + string(mat2str(colNorms,4)) + "," + ...
          """NormCorr"":" + string(mat2str(normCorr,4)) + "," + ...
          """AvgCorr"":"  + string(mat2str(avgCorr,4)) + "," + ...
          """MaxCorr"":"  + string(mat2str(maxCorr,4)) + "," + ...
          """SepScore"":" + string(mat2str(sepScore,4)) + "," + ...
          """RealH"":"    + string(mat2str(real(H),4)) + "," + ...
          """ImagH"":"    + string(mat2str(imag(H),4)) + "}";

    OutputRules = ...
        "Think step by step internally. " + ...
        "First compare ColNorm for signal strength. " + ...
        "Next compare NormCorr, AvgCorr, MaxCorr for stream separability. " + ...
        "Then consider SNR_dB and choose the ZF detection order stage by stage. " + ...
        "Return exactly one valid JSON object. " + ...
        "Do not write markdown. Do not write explanations. Do not use trailing commas. " + ...
        "Use exactly this format: {""ZF"":[a,b,c,d]}. " + ...
        "The four numbers must be unique and must contain 1,2,3,4 exactly once.";

    UserPrompt = char( ...
        "Few-shot examples (labels from 24-order full search, not greedy OSIC):" + newline + ...
        Examples + newline + newline + ...
        "Query: " + inp + newline + newline + ...
        OutputRules);

    % ── GPT API 호출 ──────────────────────────────────────
    url = 'https://api.openai.com/v1/chat/completions';

    reqBody             = struct();
    reqBody.model       = char(modelName);
    reqBody.max_tokens  = 30;
    reqBody.temperature = 0.1;
    reqBody.messages    = { ...
        struct('role','system','content', char(SystemPrompt)), ...
        struct('role','user',  'content', UserPrompt) ...
    };

    opts = weboptions( ...
        'MediaType',    'application/json', ...
        'HeaderFields', {'Authorization', ['Bearer ' char(apiKey)]; ...
                         'Content-Type',  'application/json'}, ...
        'Timeout', 3600);

    for retry = 1:maxRetry
        try
            resp    = webwrite(url, reqBody, opts);
            rawText = strtrim(string(resp.choices(1).message.content));
            answer  = extractJsonObject(rawText);
            parsed  = jsondecode(char(answer));
            zfTry   = double(parsed.ZF(:)');

            if validOrder(zfTry, Tx)
                zfOrder    = zfTry;
                isFallback = false;
                return;
            else
                error('유효하지 않은 ZF 순서값');
            end

        catch ME
            fprintf('  API 시도 %d/%d 실패: %s\n', retry, maxRetry, ME.message);
            if retry < maxRetry
                pause(retryDelay);
            end
        end
    end

    fprintf('  Fallback 사용: %s\n', mat2str(fb));
end

%% Few-shot Pool 생성 (ZF, Full-search 레이블)
function pool = buildFewShotPool_ZF(Tx, Rx, poolSize, M, labelNumSymbol, alphabet, NormFac, EsN0)
    examples = strings(1, poolSize);
    features = zeros(poolSize, 16);

    for i = 1:poolSize
        H_ex   = (randn(Rx,Tx)+1j*randn(Rx,Tx))/sqrt(2);
        osicO  = computeZF_OSICOrder(H_ex, Tx);
        zfO    = computeZF_FullSearchOrder(H_ex, Tx, M, labelNumSymbol, alphabet, NormFac, EsN0, osicO);

        [colN, corrM, avgC, maxC, sepS] = channelFeatureSummary(H_ex);
        features(i,:) = makeFeatureVector(H_ex);

        examples(i) = "{""SNR_dB"":N/A," + ...
                      """ColNorm"":"  + string(mat2str(colN,4)) + "," + ...
                      """NormCorr"":" + string(mat2str(corrM,4)) + "," + ...
                      """AvgCorr"":"  + string(mat2str(avgC,4)) + "," + ...
                      """MaxCorr"":"  + string(mat2str(maxC,4)) + "," + ...
                      """SepScore"":" + string(mat2str(sepS,4)) + "," + ...
                      """RealH"":"    + string(mat2str(real(H_ex),4)) + "," + ...
                      """ImagH"":"    + string(mat2str(imag(H_ex),4)) + "}" + ...
                      " -> {""ZF"": [" + join(string(zfO),",") + "]}";
    end

    pool = struct('examples', examples, 'features', features);
end

%% 유사도 기반 Few-shot 선택
function Examples = selectFewShot_ZF(H, pool, Nshot)
    qfeat = makeFeatureVector(H);
    Nshot = min(Nshot, numel(pool.examples));

    dist     = sum((pool.features - qfeat).^2, 2);
    [~, idx] = sort(dist, 'ascend');
    idx      = idx(1:Nshot);

    selected = pool.examples(idx);
    selected = selected(:).';
    Examples = strjoin("Example " + string(1:Nshot) + ": " + selected, newline);
end

%% 채널 특징량 계산
function [colNorms, normCorr, avgCorr, maxCorr, sepScore] = channelFeatureSummary(H)
    Tx       = size(H,2);
    colNorms = vecnorm(H,2,1);
    normCorr = zeros(Tx,Tx);

    for i = 1:Tx
        for j = 1:Tx
            normCorr(i,j) = abs(H(:,i)' * H(:,j)) / ...
                            (norm(H(:,i)) * norm(H(:,j)) + eps);
        end
    end

    avgCorr = zeros(1,Tx);
    maxCorr = zeros(1,Tx);
    for i = 1:Tx
        others      = setdiff(1:Tx,i);
        avgCorr(i)  = mean(normCorr(i,others));
        maxCorr(i)  = max(normCorr(i,others));
    end

    sepScore = colNorms ./ (avgCorr + 0.1);
end

%% 16차원 특징 벡터 생성
function feat = makeFeatureVector(H)
    [colNorms, ~, avgCorr, maxCorr, sepScore] = channelFeatureSummary(H);
    colN = colNorms ./ (norm(colNorms) + eps);
    sepS = sepScore ./ (norm(sepScore) + eps);
    feat = [colN, avgCorr, maxCorr, sepS];
end

%% JSON 추출
function answer = extractJsonObject(rawText)
    answer   = strtrim(regexprep(string(rawText), '```[a-zA-Z]*', ''));
    answer   = strrep(answer, '```', '');
    txt      = char(answer);
    startIdx = find(txt == '{', 1, 'first');
    endIdx   = find(txt == '}', 1, 'last');

    if ~isempty(startIdx) && ~isempty(endIdx) && endIdx >= startIdx
        answer = string(txt(startIdx:endIdx));
    end
    answer = strtrim(answer);
end

%% 유효 순열 검사
function tf = validOrder(o, Tx)
    tf = isnumeric(o) && numel(o)==Tx && ~any(isnan(o)) && isequal(sort(o),1:Tx);
end

%% BER Gap 계산
function gap_dB = calcBERGap(SNR_dB, BER_ref, BER_test, targetBER)
    snr_ref  = interpSNRatBER(SNR_dB, BER_ref,  targetBER);
    snr_test = interpSNRatBER(SNR_dB, BER_test, targetBER);
    if isnan(snr_ref) || isnan(snr_test)
        gap_dB = NaN;
    else
        gap_dB = snr_ref - snr_test;
    end
end

function snrAtTarget = interpSNRatBER(SNR_dB, BER, targetBER)
    snrAtTarget = NaN;
    valid = isfinite(BER) & BER > 0;
    SNR_v = SNR_dB(valid);
    BER_v = BER(valid);
    if length(SNR_v) < 2, return; end

    logBER = log10(BER_v);
    logT   = log10(targetBER);
    for k = 1:length(SNR_v)-1
        if (logBER(k)-logT)*(logBER(k+1)-logT) <= 0
            if logBER(k) == logBER(k+1)
                snrAtTarget = SNR_v(k);
            else
                snrAtTarget = SNR_v(k) + ...
                    (logT-logBER(k))*(SNR_v(k+1)-SNR_v(k))/(logBER(k+1)-logBER(k));
            end
            return;
        end
    end
end
