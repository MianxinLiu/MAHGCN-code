clear;

atlaspath='./atlas/';
outputpath='./interlayermapping/';

if ~exist(outputpath,'dir')
    mkdir(outputpath);
end

% Read atlases
cd(atlaspath)
atlas1 = MRIread('Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz');
atlas1=atlas1.vol;
ROInum1=length(unique(atlas1))-1;

atlas1_roi=zeros(ROInum1,76,61,63);
for i=1:ROInum1
    ROI=(atlas1==i);
    ROI=imresizen(double(ROI),[76/218,61/182,63/182],'nearest');
    ROI(ROI~=0)=1;
    atlas1_roi(i,:,:,:)=ROI;
end


atlas2 = MRIread('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz');
atlas2=atlas2.vol;
ROInum2=length(unique(atlas2))-1;

atlas2_roi=zeros(ROInum2,76,61,63);
for i=1:ROInum2
    ROI=(atlas2==i);
    ROI=imresizen(double(ROI),[76/218,61/182,63/182],'nearest');
    ROI(ROI~=0)=1;
    atlas2_roi(i,:,:,:)=ROI;
end

% Compute the overlapping (can define different version, e.g. based on a threshold)

%th=0.5;
mapping=zeros(ROInum1,ROInum2);
for i=1:ROInum1
    for j=1:ROInum2
        basesize=length(find(atlas1_roi(i,:,:,:)));
        overlap=squeeze(atlas1_roi(i,:,:,:))+squeeze(atlas2_roi(j,:,:,:));
        overlapsize=length(find(overlap==2));
        mapping(i,j)=overlapsize/basesize;
        %if (overlapsize/smallsize)>th
        %    mapping(i,j)=1;
        %end
    end
end

% output the mapping matrix
cd(outputpath)
save(['mapping_' num2str(ROInum1) 'to' num2str(ROInum2)],'mapping');
mapping=(mapping>0);
save(['mapping_' num2str(ROInum1) 'to' num2str(ROInum2) '_b'],'mapping');