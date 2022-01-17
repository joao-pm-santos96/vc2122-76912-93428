close all
clear all

load("RefImgs.mat")
load("IrBin.mat")
load("IrNow.mat")

ImgRes = [480 640];
nlev = 8;
corrWind = [9 9];
baseRT = 75;
%ImgRng = [800 4000];
ImgRng = [400 1000];
ImgFOV = [45.6 58.5];

windSize = corrWind(1)*corrWind(2);
ImgFOV = ImgFOV*(pi/180);

FocalLength = [ImgRes(2)/(2*tan(ImgFOV(2)/2)); ImgRes(1)/(2*tan(ImgFOV(1)/2))];

dOff_min   = ceil(baseRT*FocalLength(1)/ImgRng(1));
dOff_max   = floor(baseRT*FocalLength(1)/ImgRng(2));

numIntDisp = dOff_min - dOff_max + 1;
disp_all  = dOff_min:-1/nlev:dOff_max;
depth_all = baseRT*FocalLength(1)./disp_all;

DEPTHimg = zeros(ImgRes);

for ipix = 1:prod(ImgRes)
    % Binary window
    window_bin = IR_bin(IR_ind(:,ipix));
    
    % Noisy window
    window_now = IR_now(IR_ind(:,ipix));
    
    if sum(window_now) ~= 0
        % Estimate integer disparity with binary IR image -----------------
        snorm_ref = IR_ref(IR_ind(:,ipix),nlev,:);
        snorm_ref = logical(reshape(snorm_ref,windSize,numIntDisp));
        snorm_now = window_bin - sum(window_bin) / windSize;
        snorm_now = repmat(snorm_now,1,numIntDisp);

        % Maximize horizontal covariance
        horzCov_ref = sum(snorm_ref.*snorm_now);
        [~,dispInd] = max(horzCov_ref);
        dispLookup  = (dispInd-1)*nlev+1;

        % Sub-pixel refinement with noisy IR image ------------------------
        window_sub = IR_ref(IR_ind(:,ipix),:,dispInd);
        window_now = repmat(window_now,1,2*nlev-1);

        % Minimize sum of absolute differences
        horzCov_sub = sum(abs(window_sub-window_now));
        [~,dispInd] = min(horzCov_sub);
        dispLookup  = dispLookup + dispInd - nlev;
        
        % Convert disparity to depth from lookup table --------------------
        DEPTHimg(ipix) = depth_all(dispLookup);
    end
end

figure, imshow(DEPTHimg,[])