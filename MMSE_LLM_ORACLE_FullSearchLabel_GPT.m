clear; clc; close all;

%% ============================================================
%  4x4 MIMO 4-QAM  MMSE x SIC / OSIC / LLM-based BER 비교
%  모델: gpt-4o-mini (OpenAI)
%  구조: ZF_LLM_FullSearchLabel_GPT.m 과 동일
%      - MMSE 필터 기반 (inv(H'H + Tx/EsN0*I)*H')
%      - MMSE-OSIC ordering: post-detection SINR 기반
%      - Few-shot 라벨: 4! Full-search (MMSE 기준, OSIC tie-break)
%      - 유사도 기반 few-shot 선택 (16차원 특징 벡터)
%      - 비교: MMSE-SIC / MMSE-OSIC / MMSE-FULL / MMSE-LLM
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
EC_MMSE_SIC  = zeros(1, length(EbN0_dB));
EC_MMSE_OSIC = zeros(1, length(EbN0_dB));
EC_MMSE_FULL = zeros(1, length(EbN0_dB));
EC_MMSE_LLM  = zeros(1, length(EbN0_dB));

DoneIter    = zeros(1, length(EbN0_dB));
FallbackCnt = zeros(1, length(EbN0_dB));

MMSE_MatchCount      = zeros(1, length(EbN0_dB));   % LLM vs OSIC full match
MMSE_PartialMatch    = zeros(1, length(EbN0_dB));    % LLM vs OSIC 1st match
MMSE_FullRefMatch    = zeros(1, length(EbN0_dB));    % LLM vs FULL full match
MMSE_FullRef1stMatch = zeros(1, length(EbN0_dB));    % LLM vs FULL 1st match

BER_conv_MMSE_SIC  = nan(length(EbN0_dB), NumberIteration);
BER_conv_MMSE_LLM  = nan(length(EbN0_dB), NumberIteration);
BER_conv_MMSE_OSIC = nan(length(EbN0_dB), NumberIteration);
BER_conv_MMSE_FULL = nan(length(EbN0_dB), NumberIteration);

%% ── Few-shot 예시 후보 pool 생성 ─────────────────────────
poolFile = sprintf('ExamplePool_MMSE_GPT_L%d.mat', LabelNumSymbol);

if exist(poolFile, 'file')
    fprintf('기존 pool 로드 중: %s\n', poolFile);
    loaded = load(poolFile, 'ExamplePool', 'LabelNumSymbol');
    if loaded.LabelNumSymbol == LabelNumSymbol
        ExamplePool = loaded.ExamplePool;
        fprintf('로드 완료\n\n');
    else
        fprintf('LabelNumSymbol 불일치 → pool 재생성\n');
        ExamplePool = generatePool(Tx, Rx, EbN0_dB, EsN0, ExamplePoolSize, M, LabelNumSymbol, alphabet, NormFac);
        save(poolFile, 'ExamplePool', 'LabelNumSymbol');
        fprintf('완료 및 저장\n\n');
    end
else
    fprintf('Few-shot 후보 pool 생성 중...\n');
    ExamplePool = repmat(struct('examples', strings(1, ExamplePoolSize), ...
                                'features', zeros(ExamplePoolSize, 16)), ...
                         1, length(EbN0_dB));
    % MMSE ordering은 SNR에 따라 달라지므로 SNR별로 pool을 생성
    for i = 1:length(EbN0_dB)
        ExamplePool(i) = buildFewShotPool_MMSE(Tx, Rx, EbN0_dB(i), EsN0(i), ExamplePoolSize, M, LabelNumSymbol, alphabet, NormFac);
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

    for iTotal = 1:NumberIteration

        H = (randn(Rx,Tx) + 1j*randn(Rx,Tx)) ./ sqrt(2);

        fixedOrder      = 1:Tx;
        mmseOSICOrder   = computeMMSE_OSICOrder(H, cur_EsN0, NormFac);
        allOrders_24    = perms(1:Tx);          % 24×4
        err_allOrders   = zeros(1, size(allOrders_24,1));  % oracle 누적용

        % 현재 H와 특징이 가까운 few-shot 예시 선택
        Examples = selectFewShot_MMSE(H, ExamplePool(idxSNR), Nshot);

        [mmseLLM, ~, isFB] = getLLMOrder_MMSE( ...
            H, EbN0_dB(idxSNR), Examples, ...
            apiKey, modelName, maxRetry, retryDelay);

        if isFB
            FallbackCnt(idxSNR) = FallbackCnt(idxSNR) + 1;
        end

        if isequal(mmseLLM, mmseOSICOrder)
            MMSE_MatchCount(idxSNR) = MMSE_MatchCount(idxSNR) + 1;
        end
        if mmseLLM(1) == mmseOSICOrder(1)
            MMSE_PartialMatch(idxSNR) = MMSE_PartialMatch(idxSNR) + 1;
        end

        fprintf('  MMSE-OSIC=%s\n', mat2str(mmseOSICOrder));
        fprintf('  MMSE-LLM =%s  FB=%d\n', mat2str(mmseLLM), isFB);

        for idxFrame = 1:NumSymbolPerChannel
            BitSeq = randi([0 1], 1, LengthBit);

            SigSeq = zeros(1, Tx);
            for k = 1:Tx
                SigSeq(k) = bi2de(BitSeq((k-1)*nu+1:k*nu), 'left-msb');
            end

            Symbol = qammod(SigSeq.', M) / NormFac;
            Noise  = (randn(Rx,1) + 1j*randn(Rx,1)) / sqrt(2);
            y      = H * Symbol + Noise * sqrt(1/cur_EsN0);

            EC_MMSE_SIC(idxSNR)  = EC_MMSE_SIC(idxSNR)  + MMSE_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,cur_EsN0,fixedOrder);
            EC_MMSE_OSIC(idxSNR) = EC_MMSE_OSIC(idxSNR) + MMSE_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,cur_EsN0,mmseOSICOrder);
            EC_MMSE_LLM(idxSNR)  = EC_MMSE_LLM(idxSNR)  + MMSE_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,cur_EsN0,mmseLLM);

            % Oracle FULL: 같은 symbol/noise로 24개 전수조사 → 누적
            for oi = 1:size(allOrders_24,1)
                err_allOrders(oi) = err_allOrders(oi) + ...
                    MMSE_SIC_detect(H,y,SigSeq,BitSeq,M,NormFac,alphabet,cur_EsN0,allOrders_24(oi,:));
            end
        end

        % Oracle FULL: 채널당 최소 누적 error → 정의상 FULL ≤ OSIC 보장
        [minFullErr, bestOI] = min(err_allOrders);
        EC_MMSE_FULL(idxSNR) = EC_MMSE_FULL(idxSNR) + minFullErr;
        mmseFullOrder = allOrders_24(bestOI,:);   % 일치율 통계용

        if isequal(mmseLLM, mmseFullOrder)
            MMSE_FullRefMatch(idxSNR) = MMSE_FullRefMatch(idxSNR) + 1;
        end
        if mmseLLM(1) == mmseFullOrder(1)
            MMSE_FullRef1stMatch(idxSNR) = MMSE_FullRef1stMatch(idxSNR) + 1;
        end
        fprintf('  MMSE-FULL=%s  (oracle, same eval symbols)\n', mat2str(mmseFullOrder));

        DoneIter(idxSNR) = DoneIter(idxSNR) + 1;

        elapsed   = toc(startTime);
        doneTotal = sum(DoneIter);
        totalCalls= length(EbN0_dB) * NumberIteration;
        remaining = (elapsed / doneTotal) * (totalCalls - doneTotal);

        fprintf('  iter %d/%d 완료 | SNR경과 %.1f분 | 전체경과 %.1f분 | 남은시간 약 %.1f분\n', ...
            DoneIter(idxSNR), NumberIteration, toc(snrStartTime)/60, elapsed/60, remaining/60);

        tb_now = LengthBit * NumSymbolPerChannel * DoneIter(idxSNR);
        BER_conv_MMSE_SIC(idxSNR,  DoneIter(idxSNR)) = EC_MMSE_SIC(idxSNR)  / tb_now;
        BER_conv_MMSE_LLM(idxSNR,  DoneIter(idxSNR)) = EC_MMSE_LLM(idxSNR)  / tb_now;
        BER_conv_MMSE_OSIC(idxSNR, DoneIter(idxSNR)) = EC_MMSE_OSIC(idxSNR) / tb_now;
        BER_conv_MMSE_FULL(idxSNR, DoneIter(idxSNR)) = EC_MMSE_FULL(idxSNR) / tb_now;

        save('progress_MMSE_FullSearchLabel_GPT.mat', ...
            'EC_MMSE_SIC','EC_MMSE_OSIC','EC_MMSE_FULL','EC_MMSE_LLM', ...
            'DoneIter','FallbackCnt', ...
            'MMSE_MatchCount','MMSE_PartialMatch','MMSE_FullRefMatch','MMSE_FullRef1stMatch', ...
            'BER_conv_MMSE_SIC','BER_conv_MMSE_LLM','BER_conv_MMSE_OSIC','BER_conv_MMSE_FULL', ...
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

BER_MMSE_SIC  = nan(size(EbN0_dB));
BER_MMSE_OSIC = nan(size(EbN0_dB));
BER_MMSE_FULL = nan(size(EbN0_dB));
BER_MMSE_LLM  = nan(size(EbN0_dB));

BER_MMSE_SIC(valid)  = EC_MMSE_SIC(valid)  ./ TotalBits(valid);
BER_MMSE_OSIC(valid) = EC_MMSE_OSIC(valid) ./ TotalBits(valid);
BER_MMSE_FULL(valid) = EC_MMSE_FULL(valid) ./ TotalBits(valid);
BER_MMSE_LLM(valid)  = EC_MMSE_LLM(valid)  ./ TotalBits(valid);

%% ── 순서 일치율 계산 ─────────────────────────────────────
MMSE_FullMatchRate   = nan(size(EbN0_dB));
MMSE_FirstMatchRate  = nan(size(EbN0_dB));
MMSE_FullRefMatchRate= nan(size(EbN0_dB));
MMSE_FullRefFirstRate= nan(size(EbN0_dB));

MMSE_FullMatchRate(valid)    = MMSE_MatchCount(valid)      ./ DoneIter(valid) * 100;
MMSE_FirstMatchRate(valid)   = MMSE_PartialMatch(valid)    ./ DoneIter(valid) * 100;
MMSE_FullRefMatchRate(valid) = MMSE_FullRefMatch(valid)    ./ DoneIter(valid) * 100;
MMSE_FullRefFirstRate(valid) = MMSE_FullRef1stMatch(valid) ./ DoneIter(valid) * 100;

%% ── BER Gap 계산 (보간) ──────────────────────────────────
targetBER = 1e-2;
MMSE_LLM_Gain_vs_SIC_dB  = calcBERGap(EbN0_dB, BER_MMSE_SIC,  BER_MMSE_LLM,  targetBER);
MMSE_LLM_Loss_vs_OSIC_dB = calcBERGap(EbN0_dB, BER_MMSE_LLM,  BER_MMSE_OSIC, targetBER);
MMSE_LLM_Loss_vs_FULL_dB = calcBERGap(EbN0_dB, BER_MMSE_LLM,  BER_MMSE_FULL, targetBER);

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
            EbN0_dB(i), MMSE_FullMatchRate(i), MMSE_FirstMatchRate(i));
    end
end

fprintf('\n===== LLM vs FULL-search 순서 일치율 =====\n');
fprintf('%-8s  %-16s  %-16s\n', 'Eb/N0','FULL_Full(%)','FULL_1st(%)');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d  %-16.1f  %-16.1f\n', ...
            EbN0_dB(i), MMSE_FullRefMatchRate(i), MMSE_FullRefFirstRate(i));
    end
end

fprintf('\n===== BER Gap (targetBER=%.0e) =====\n', targetBER);
fprintf('MMSE: LLM Gain vs SIC  = %.3f dB\n', MMSE_LLM_Gain_vs_SIC_dB);
fprintf('MMSE: LLM Loss vs OSIC = %.3f dB\n', MMSE_LLM_Loss_vs_OSIC_dB);
fprintf('MMSE: LLM Loss vs FULL = %.3f dB\n', MMSE_LLM_Loss_vs_FULL_dB);

fprintf('\n===== 정확한 BER 값 =====\n');
fprintf('%-8s %-14s %-14s %-14s %-14s\n', 'Eb/N0','MMSE-SIC','MMSE-OSIC','MMSE-FULL (oracle)','MMSE-LLM');
for i = 1:length(EbN0_dB)
    if valid(i)
        fprintf('%-8d %-14.6e %-14.6e %-14.6e %-14.6e\n', ...
            EbN0_dB(i), BER_MMSE_SIC(i), BER_MMSE_OSIC(i), BER_MMSE_FULL(i), BER_MMSE_LLM(i));
    end
end

%% ── 저장 ─────────────────────────────────────────────────
save('final_MMSE_FullSearchLabel_GPT.mat', 'EbN0_dB','DoneIter','FallbackCnt', ...
    'BER_MMSE_SIC','BER_MMSE_OSIC','BER_MMSE_FULL','BER_MMSE_LLM', ...
    'MMSE_FullMatchRate','MMSE_FirstMatchRate', ...
    'MMSE_FullRefMatchRate','MMSE_FullRefFirstRate', ...
    'MMSE_LLM_Gain_vs_SIC_dB','MMSE_LLM_Loss_vs_OSIC_dB','MMSE_LLM_Loss_vs_FULL_dB', ...
    'Nshot','ExamplePoolSize','LabelNumSymbol');

%% ── 그래프 1: BER 곡선 ──────────────────────────────────
floorBER = 1e-5;
toPlot   = @(b) max(b, floorBER);

figure;
semilogy(EbN0_dB, toPlot(BER_MMSE_SIC),  '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
semilogy(EbN0_dB, toPlot(BER_MMSE_OSIC), '-s', 'LineWidth',1.5, 'MarkerSize',7);
semilogy(EbN0_dB, toPlot(BER_MMSE_FULL), '-d', 'LineWidth',1.5, 'MarkerSize',7);
semilogy(EbN0_dB, toPlot(BER_MMSE_LLM),  '-^', 'LineWidth',2.0, 'MarkerSize',8);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('BER');
title('MMSE-only: Oracle Full-search + LLM Ordering (GPT-4o-mini)');
legend('MMSE-SIC','MMSE-OSIC','MMSE-FULL (oracle)','MMSE-LLM','Location','southwest');
ylim([floorBER 1]);

%% ── 그래프 2: 순서 일치율 ───────────────────────────────
figure;
plot(EbN0_dB, MMSE_FullMatchRate,   '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
plot(EbN0_dB, MMSE_FirstMatchRate,  '-s', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, MMSE_FullRefMatchRate,'-^', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, MMSE_FullRefFirstRate,'-d', 'LineWidth',1.5, 'MarkerSize',7);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('Match Rate (%)');
title('MMSE LLM 순서 일치율 (GPT-4o-mini)');
legend('vs OSIC 완전일치','vs OSIC 1번째일치','vs FULL 완전일치','vs FULL 1번째일치','Location','best');
ylim([0 100]);

%% ── 그래프 3: BER Gap ───────────────────────────────────
MMSE_gap_vs_sic  = nan(size(EbN0_dB));
MMSE_gap_vs_osic = nan(size(EbN0_dB));
MMSE_gap_vs_full = nan(size(EbN0_dB));

for i = 1:length(EbN0_dB)
    if valid(i) && BER_MMSE_LLM(i)>0 && BER_MMSE_SIC(i)>0 && BER_MMSE_OSIC(i)>0 && BER_MMSE_FULL(i)>0
        MMSE_gap_vs_sic(i)  = 10*log10(BER_MMSE_SIC(i)  / BER_MMSE_LLM(i));
        MMSE_gap_vs_osic(i) = 10*log10(BER_MMSE_LLM(i)  / BER_MMSE_OSIC(i));
        MMSE_gap_vs_full(i) = 10*log10(BER_MMSE_LLM(i)  / BER_MMSE_FULL(i));
    end
end

figure;
plot(EbN0_dB, MMSE_gap_vs_sic,  '-o', 'LineWidth',1.5, 'MarkerSize',7); hold on;
plot(EbN0_dB, MMSE_gap_vs_osic, '-s', 'LineWidth',1.5, 'MarkerSize',7);
plot(EbN0_dB, MMSE_gap_vs_full, '-^', 'LineWidth',1.5, 'MarkerSize',7);
yline(0, 'k--', 'LineWidth',1.0);
grid on;
xlabel('Eb/N0 [dB]');
ylabel('BER Ratio Gap (dB)');
title('MMSE BER Gap: vs SIC 양수=개선, vs OSIC/FULL 양수=손실 (GPT-4o-mini)');
legend('MMSE: LLM Gain vs SIC','MMSE: LLM Loss vs OSIC','MMSE: LLM Loss vs FULL','Location','best');

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
    semilogy(iter_axis, BER_conv_MMSE_SIC(snr_idx,1:n_done),  '-o', 'LineWidth',1.5, 'MarkerSize',4); hold on;
    semilogy(iter_axis, BER_conv_MMSE_OSIC(snr_idx,1:n_done), '-s', 'LineWidth',1.5, 'MarkerSize',4);
    semilogy(iter_axis, BER_conv_MMSE_FULL(snr_idx,1:n_done), '-d', 'LineWidth',1.5, 'MarkerSize',4);
    semilogy(iter_axis, BER_conv_MMSE_LLM(snr_idx,1:n_done),  '-^', 'LineWidth',2.0, 'MarkerSize',5);
    grid on;
    xlabel('Iteration 수');
    ylabel('누적 BER');
    title('MMSE BER 수렴 과정 (Eb/N0 = 10 dB, GPT-4o-mini)');
    legend('MMSE-SIC','MMSE-OSIC','MMSE-FULL (oracle)','MMSE-LLM','Location','best');
end

%% ============================================================
%  함수 정의
%% ============================================================

%% MMSE-SIC 검출 함수
function bec = MMSE_SIC_detect(H, y, SigSeq, BitSeq, M, NormFac, alphabet, EsN0, order)
    Tx     = size(H,2);
    nu     = log2(M);
    H_rem  = H;
    y_rem  = y;
    remIdx = 1:Tx;
    detSig = zeros(1,Tx);

    for stage = 1:Tx
        target = order(stage);
        rel    = find(remIdx == target);
        n_rem  = size(H_rem,2);

        % MMSE 필터: W = NormFac * inv(H'H + n_rem/EsN0*I) * H'
        A = H_rem'*H_rem + n_rem/EsN0 * eye(n_rem);
        W = NormFac * (A \ H_rem');
        z = W(rel,:) * y_rem;

        detSig(target) = qamdemod(z, M);
        remod  = alphabet(detSig(target)+1);
        y_rem  = y_rem - H_rem(:,rel) * remod;

        H_rem(:,rel) = [];
        remIdx(rel)  = [];
    end

    detBits = zeros(1, Tx*nu);
    for k = 1:Tx
        detBits((k-1)*nu+1:k*nu) = de2bi(detSig(k), nu, 'left-msb');
    end
    bec = sum(BitSeq ~= detBits);
end

%% MMSE-OSIC 순서 계산 (post-detection SINR 기반)
function order = computeMMSE_OSICOrder(H, EsN0, NormFac)
    Tx     = size(H,2);
    snr    = EsN0 / NormFac^2;
    order  = zeros(1,Tx);
    H_rem  = H;
    remIdx = 1:Tx;

    for stage = 1:Tx
        n_rem = size(H_rem,2);
        A     = H_rem'*H_rem + n_rem/EsN0 * eye(n_rem);
        W     = NormFac * (A \ H_rem');

        WH2  = abs(W * H_rem).^2;
        sinr = snr * diag(WH2) ./ (snr*(sum(WH2,2)-diag(WH2)) + sum(abs(W).^2,2));

        [~, rel]     = max(sinr);
        order(stage) = remIdx(rel);

        H_rem(:,rel) = [];
        remIdx(rel)  = [];
    end
end

%% MMSE Full-Search 최적 순서 계산 (probe 심볼 기반, OSIC tie-break)
function [bestOrder, errByOrder] = computeMMSE_FullSearchOrder(H, EsN0, NormFac, M, numProbeSym, alphabet, tieBreakOrder)
    Tx        = size(H,2);
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
            errByOrder(oi) = errByOrder(oi) + MMSE_SIC_detect( ...
                H, y, SigSeq, BitSeq, M, NormFac, alphabet, EsN0, allOrders(oi,:));
        end
    end

    minErr  = min(errByOrder);
    candIdx = find(errByOrder == minErr);
    bestIdx = candIdx(1);

    if nargin >= 7 && ~isempty(tieBreakOrder)
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
function [mmseOrder, answer, isFallback] = getLLMOrder_MMSE( ...
        H, snr_dB, Examples, apiKey, modelName, maxRetry, retryDelay)

    Tx = size(H,2);

    % Fallback: SepScore 기반
    [~, ~, ~, ~, sepScore] = channelFeatureSummary(H);
    [~, fb] = sort(sepScore, 'descend');
    mmseOrder    = fb;
    answer     = "";
    isFallback = true;

    % ── System Prompt ──────────────────────────────────────
    SystemPrompt = ...
        "You are an expert in MIMO signal processing using MMSE equalization. " + ...
        "Your task is to determine the optimal stream detection order for MMSE-SIC " + ...
        "(Minimum Mean Square Error Successive Interference Cancellation)." + newline + newline + ...
        "MMSE perspective:" + newline + ...
        "MMSE equalization balances interference suppression and noise amplification. " + ...
        "A reliable stream has strong channel energy AND good separation from remaining columns, " + ...
        "while also considering the noise level relative to SNR." + newline + newline + ...
        "Input features explained:" + newline + ...
        "- SNR_dB: Signal-to-noise ratio in dB. MMSE ordering depends on SNR." + newline + ...
        "- ColNorm: Euclidean norm of each column of H (stream signal strength)." + newline + ...
        "- NormCorr: Normalized absolute correlation between channel columns (4x4 matrix). " + ...
        "  Lower off-diagonal value = better stream separability." + newline + ...
        "- AvgCorr: Average normalized correlation of each stream with all other streams." + newline + ...
        "- MaxCorr: Maximum normalized correlation of each stream with any other stream." + newline + ...
        "- SepScore: Simple separability indicator = ColNorm / (AvgCorr + 0.1). Larger = more separable." + newline + ...
        "- RealH, ImagH: Real and imaginary parts of channel matrix H (4x4 MATLAB format)." + newline + newline + ...
        "Detection order rules:" + newline + ...
        "1. The first detected stream is most important due to SIC error propagation." + newline + ...
        "2. Detect the most reliable stream first (strong ColNorm AND low AvgCorr/MaxCorr)." + newline + ...
        "3. Do not rely only on ColNorm. High correlation causes interference under MMSE." + newline + ...
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
        "Then consider SNR_dB and choose the MMSE detection order stage by stage. " + ...
        "Return exactly one valid JSON object. " + ...
        "Do not write markdown. Do not write explanations. Do not use trailing commas. " + ...
        "Use exactly this format: {""MMSE"":[a,b,c,d]}. " + ...
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
            mmseTry   = double(parsed.MMSE(:)');

            if validOrder(mmseTry, Tx)
                mmseOrder    = mmseTry;
                isFallback = false;
                return;
            else
                error('유효하지 않은 MMSE 순서값');
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

%% Few-shot Pool 생성 (MMSE, Full-search 레이블)
function pool = buildFewShotPool_MMSE(Tx, Rx, snr_dB, EsN0_val, poolSize, M, labelNumSymbol, alphabet, NormFac)
    examples = strings(1, poolSize);
    features = zeros(poolSize, 16);

    for i = 1:poolSize
        H_ex  = (randn(Rx,Tx)+1j*randn(Rx,Tx))/sqrt(2);
        osicO = computeMMSE_OSICOrder(H_ex, EsN0_val, NormFac);
        mmseO = computeMMSE_FullSearchOrder(H_ex, EsN0_val, NormFac, M, labelNumSymbol, alphabet, osicO);

        [colN, corrM, avgC, maxC, sepS] = channelFeatureSummary(H_ex);
        features(i,:) = makeFeatureVector(H_ex);

        examples(i) = "{""SNR_dB"":" + string(snr_dB) + "," + ...
                      """ColNorm"":"  + string(mat2str(colN,4)) + "," + ...
                      """NormCorr"":" + string(mat2str(corrM,4)) + "," + ...
                      """AvgCorr"":"  + string(mat2str(avgC,4)) + "," + ...
                      """MaxCorr"":"  + string(mat2str(maxC,4)) + "," + ...
                      """SepScore"":" + string(mat2str(sepS,4)) + "," + ...
                      """RealH"":"    + string(mat2str(real(H_ex),4)) + "," + ...
                      """ImagH"":"    + string(mat2str(imag(H_ex),4)) + "}" + ...
                      " -> {""MMSE"": [" + join(string(mmseO),",") + "]}";
    end

    pool = struct('examples', examples, 'features', features);
end

%% 유사도 기반 Few-shot 선택
function Examples = selectFewShot_MMSE(H, pool, Nshot)
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
