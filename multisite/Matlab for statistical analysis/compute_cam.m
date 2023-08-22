clear;
cd('/media/user/Elements/all_disease_project/gradcam');

gradcamdata=zeros(500,5);
rois=[500,400,300,200,100];
load('/home/user/Desktop/all_disease/results/ave_performance_multi.mat')
weight=ave_multi(4,:);


% normalize within each site and compute
for theset=1:6

    gradcamdata=zeros(500,5);

    for i=1:5
        temp=[];
        load(['_gradcam_down' num2str(i) '.mat'])
        mask_all(mask_all<0)=0;

        for cv = 1:10
            load(['cv_' num2str(cv) '_0vs1_predictions.mat'])
            pos=find((predicted_all==1) & (test_y_all == 1) & (site==theset));
            cam=mean(mask_all(pos,:,cv))';
            cam=normalize(cam,1,'range');
            temp = [temp,weight(cv)*cam];
        end
        data=mean(temp,2);
        data=normalize(data,1,'range');

        gradcamdata(1:rois(i),5-i+1)=data;
    end

    save(['gradcamdata_site' num2str(theset) '.mat'],'gradcamdata');
end

% all site ave

gradcamdata_all=zeros(500,5);
for theset=1:6
    load(['gradcamdata_site' num2str(theset) '.mat']);
    gradcamdata_all=gradcamdata_all+gradcamdata;
end
gradcamdata=gradcamdata_all/6;

gradcamdata_all=zeros(500,5);
for theset=[2,3,4]
    load(['gradcamdata_site' num2str(theset) '.mat']);
    gradcamdata_all=gradcamdata_all+gradcamdata;
end
gradcamdata=gradcamdata_all/3;

% brain map visualization
clear;
atlas_L = gifti('D:\HKBU\MMPatlas\L.annotation.label.gii');
atlas_R = gifti('D:\HKBU\MMPatlas\R.annotation.label.gii');
surf_L=gifti('D:\HKBU\MMPatlas\Q1-Q6_RelatedParcellation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii');
surf_R=gifti('D:\HKBU\MMPatlas\Q1-Q6_RelatedParcellation210.R.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii');

RSNpath='D:\Shanghai Tech\multiresolution project\MSforjournal\figure\gradcam\cifti\';
savepath='D:\Shanghai Tech\commondisease project\figure\';

task='all';
load(['D:\Shanghai Tech\commondisease project\results\gradcamdata_' task '.mat']);
% pos=gradcamdata~=0;
% gradcamdata(pos)=log10(gradcamdata(pos));
for ROI=100:100:500
    RSN=ft_read_cifti([RSNpath 'Schaefer2018_' num2str(ROI) 'Parcels_7Networks_order.dscalar.nii']);

    %1:32492 left; 32492:end right
    atlas_L.cdata=RSN.dscalar(1:32492);
    atlas_R.cdata=RSN.dscalar(32493:end);

    show_L=atlas_L;
    show_L.cdata=show_L.cdata*0;
    subcor_L=atlas_L;
    subcor_L.cdata(subcor_L.cdata~=0)=nan;
    show_R=atlas_R;
    show_R.cdata=show_R.cdata*0;
    subcor_R=atlas_R;
    subcor_R.cdata(subcor_R.cdata~=0)=nan;
    for i=1:(ROI/2)
        temp=[];
        pos=find(atlas_L.cdata==i);
        show_L.cdata(pos)=gradcamdata(i,ROI/100);
        temp=[];
        pos=find(atlas_R.cdata==i+(ROI/2));
        show_R.cdata(pos)=gradcamdata(i+(ROI/2),ROI/100);
    end
    %show_L.cdata(atlas_L.cdata==0)=1;
    %show_R.cdata(atlas_R.cdata==0)=1;
    figure;
    plot(surf_L,show_L);
    colormap('parula');
    caxis([0,0.15])

%     caxis([min(gradcamdata(:,ROI/100)),max(gradcamdata(:,ROI/100))])
    savefig([savepath task num2str(ROI) '_L.fig'])
    ax=gca;
    saveas(ax,[savepath task num2str(ROI) '_L_1.tif'])
    ax.CameraPosition=[-1.6153   -0.0175    0.3233]*10^3;
    saveas(ax,[savepath task num2str(ROI) '_L_2.tif'])
    ax.CameraPosition=[1.5804   -0.0175    0.0437]*10^3;
    camlight(-40,-10);
    hold on;
    plot(surf_L,subcor_L);
    ax2=gca;
    colormap(ax2,'gray');
    ax2.CameraPosition=[1.5804   -0.0175    0.0437]*10^3;
    camlight(-40,-10);
    saveas(ax,[savepath task num2str(ROI) '_L_3.tif'])
    close;
    
    figure;
    plot(surf_R,show_R);
    colormap('parula');
    caxis([min(gradcamdata(:,ROI/100)),max(gradcamdata(:,ROI/100))])
    ax=gca;
    saveas(ax,[savepath task num2str(ROI) '_R_1.tif'])
%     ax.CameraPosition=[-1.5309   -0.3204    0.0712]*10^3;
    ax.CameraPosition=[-1.5432   -0.2548    0.0156]*10^3;
    hold on;
    plot(surf_R,subcor_R);
    ax2=gca;
    colormap(ax2,'gray');
    ax2.CameraPosition=[-1.5432   -0.2548    0.0156]*10^3;
    saveas(ax,[savepath task num2str(ROI) '_R_3.tif'])
    close;
    figure;
    plot(surf_R,show_R);
    colormap('parula');
%     caxis([min(gradcamdata(:,ROI/100)),max(gradcamdata(:,ROI/100))])
    caxis([0,0.15])
    ax=gca;
    ax.CameraPosition=[1.6093    0.0662    0.2374]*10^3;
    camlight(-40,-10);
    saveas(ax,[savepath task num2str(ROI) '_R_2.tif'])
    close;
end

savepath='D:\Shanghai Tech\commondisease project\figure\combine\';
cd('D:\Shanghai Tech\commondisease project\figure\');
task='dep';
for scale=100:100:500
X = imread([task num2str(scale) '_L_2.tif']);
ax1 = axes('Position',[0.1 0.60 0.3 0.3]);
X=imcrop(X, [160   90  580  435]);
imshow(X);
axis off
X = imread([task num2str(scale) '_L_3.tif']);
ax1 = axes('Position',[0.1 0.15 0.3 0.3]);
X=imcrop(X, [160   90  580  435]);
imshow(X);
axis off
X = imread([task num2str(scale) '_R_2.tif']);
ax1 = axes('Position',[0.60 0.60 0.3 0.3]);
X=imcrop(X, [160   90  580  435]);
imshow(X);
axis off
X = imread([task num2str(scale) '_R_3.tif']);
ax1 = axes('Position',[0.60 0.15 0.3 0.3]);
X=imcrop(X, [160   90  580  435]);
imshow(X);
axis off
X = imread([task num2str(scale) '_L_1.tif']);
ax1 = axes('Position',[0.27 0.35 0.33 0.33]);
X = imcrop(X,[280 20 340 600]);
imshow(X);
axis off
X = imread([task num2str(scale) '_R_1.tif']);
ax1 = axes('Position',[0.39 0.35 0.33 0.33]);
X = imcrop(X,[280 20 340 600]);
imshow(X);
axis off
annotation('textbox',[0.04,0.5,0.1,0.1],'String','L','LineStyle','none','Fontsize',30)
annotation('textbox',[0.90,0.5,0.1,0.1],'String','R','LineStyle','none','Fontsize',30)
annotation('textbox',[0.38,0.9,0.1,0.1],'String',[num2str(scale) ' ROIs'],'LineStyle','none','Fontsize',20)
saveas(gca,[savepath task num2str(scale) '.tif'])
close;
end

% bar plot
figure;
col=[120,18,134;70,130,180;0,118,14;196,58,250;220,248,164;230,148,34;205,62,78]/256;
name={'VIS','SM','ATT','SAL','LIM','FP','DMN'};
titlename={'100 ROI','200 ROI','300 ROI','400 ROI','500 ROI'};
load('par.mat');
meancam=zeros(5,7);
for i=1:5
    for j=1:7
        pos=find(par(:,i)==j);
        meancam(i,j)=mean(gradcamdata(pos,i));
    end
end

for j=1:5
    subplot(1,5,j)
    for i=1:7
        bar(i,meancam(j,i),'FaceColor',col(i,:),'EdgeColor','none')
        hold on;
    end
    title(titlename{j})
    ylabel('Averaged CAM rank')
    xticks([1:7])
    xticklabels(name);
    xtickangle(35)
%     ylim([0,1])
    set(gca,'FontSize',12);
end