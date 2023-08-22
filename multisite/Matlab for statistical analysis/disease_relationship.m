clear;

load('/home/user/Desktop/all_disease/results/ave_performance_multi.mat')
% weight=[0.67,0.71,0.7, 0.7,0.71,0.68, 0.71,0.68,0.71,0.65];
weight=ave_multi(4,:);
% correlation distance
disM=zeros(4410,4410,3);

for cv=1:10
    load(['cv_' num2str(cv) '_0vs1_feature.mat']);
    disM(:,:,1)=disM(:,:,1)+weight(cv)*(1-corrcoef(feature2_all'));
    disM(:,:,2)=disM(:,:,2)+weight(cv)*(1-corrcoef(feature3_all'));
    disM(:,:,3)=disM(:,:,3)+weight(cv)*(1-corrcoef(feature4_all'));
end
disM = disM/10;

load('label.mat')

%     V = cmdscale(disM(:,:,l));
[V,lambda] = diffusion_mapping(disM(:,:,3), 3, 0.5, 0);

figure;

for i=[0,1,3,4,5]
%         scatter3(V(label==i,1),V(label==i,2),V(label==i,3),'filled');
	scatter(V(label==i,1),V(label==i,2),'filled');
	alpha(0.5)
	hold on
end
xlabel('Gradient 1')
ylabel('Gradient 2')
%     zlabel('MDS axis3')
legend('HC','ASD','VCI','MCI/AD','ADHD')


% draw disM (all)
thedisM=squeeze(disM(:,:,3));
datasizes=[1011, 297, 267, 1350, 717, 768];
databin=cumsum(datasizes);

imagesc(thedisM)
[x,y]=meshgrid([0,databin],[0,databin]);
hold on 
plot(x,y,'k-',x',y','k-','LineWidth',2)
xticks([datasizes(1)/2, databin(1:5)+diff(databin)/2])
xtickangle(45)
xticklabels({'ABIDE','RENJI','HUASHAN','ADNI','OASIS','ADHD-200'})
yticks([1011/2, databin(1:5)+diff(databin)/2])
yticklabels({'ABIDE','RENJI','HUASHAN','ADNI','OASIS','ADHD-200'})
set(gca, 'FontSize', 15)

% draw disM (BD)
selectpos=[find(label==1);find(label==3);find(label==4);find(label==5)];
newdisM=thedisM(selectpos,selectpos);
imagesc(newdisM)

datasizes=[length(find(label==1)),length(find(label==3)),length(find(label==4)),length(find(label==5))];
databin=cumsum(datasizes);

imagesc(newdisM)
[x,y]=meshgrid([0.5,databin],[0.5,databin]);
hold on 
plot(x,y,'k-',x',y','k-','LineWidth',2)
xticks([datasizes(1)/2, databin(1:3)+diff(databin)/2])
xtickangle(45)
xticklabels({'ASD','VCI','MCI/AD','ADHD'})
yticks([datasizes(1)/2, databin(1:3)+diff(databin)/2])
yticklabels({'ASD','VCI','MCI/AD','ADHD'})
set(gca, 'FontSize', 15)

% distance among diseases
DistbDis=zeros(4,4);
thelabel=[1,3,4,5];
for i=1:4
    for j=1:4
     idx_s=find(label==thelabel(i));idx_t=find(label==thelabel(j));
     block= thedisM(idx_s, idx_t);
     block(block==0)=[];
     DistbDis(i,j) = mean(mean(block));
    end
end

imagesc(DistbDis)
colormap('jet')
caxis([0.2,0.55])
[x,y]=meshgrid([0.5:4.5],[0.5:4.5]);
hold on 
plot(x,y,'k-',x',y','k-','LineWidth',2)
xticks([1,2,3,4])
xtickangle(45)
xticklabels({'ASD','VCI','MCI/AD','ADHD'})
yticks([1,2,3,4])
yticklabels({'ASD','VCI','MCI/AD','ADHD'})
set(gca, 'FontSize', 15)


% ADNI demonstration
% AD 1576-1708 eMCI1709-2052 lmci 2053 2360  hc  2361 2925

% V = cmdscale(disM(:,:,3));
[V,lambda] = diffusion_mapping(disM(:,:,3), 3, 0.5, 0);

scatter(V(2361:2925,1),V(2361:2925,2),'filled');
alpha(0.5)
hold on
scatter(V(1709:2052,1),V(1709:2052,2),'filled');
alpha(0.5)
scatter(V(2053:2360,1),V(2053:2360,2),'filled');
alpha(0.5)
scatter(V(1576:1708,1),V(1576:1708,2),'filled');
alpha(0.5)
scatter(V(label==3,1),V(label==3,2),'filled');


xlabel('Graident 1')
ylabel('Gradient 2')
legend('HC','eMCI','lMCI','AD')


s=1; % s=2
DATA.d1=V(2361:2925,s); DATA.d2=V(1709:2052,s);DATA.d3=V(2053:2360,s);
DATA.d4=V(1576:1708,s); DATA.d5=V(label==3,s);
vio=violinplot(DATA);
vio(1).ViolinColor=[0.65,0.65,0.65];
vio(2).ViolinColor=[0.30,0.75,0.93];
vio(3).ViolinColor=[0.0,0.45,0.74];
vio(4).ViolinColor=[1.0,0.0,0.0];
vio(5).ViolinColor=[0.93,0.69,0.13];

xlim([0.5,4.5])
ylim([-0.8,0.8])
xticks([1,2,3,4])
xticklabels({'HC','eMCI','lMCI','AD'})
ylabel('Gradient 2')

hold on;
sigstar({[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]},[ranksum(DATA.d1,DATA.d2),ranksum(DATA.d1,DATA.d3),ranksum(DATA.d1,DATA.d4),...
   ranksum(DATA.d2,DATA.d3),ranksum(DATA.d2,DATA.d4),ranksum(DATA.d3,DATA.d4)]);


% BD distribution in gradients
gra=2;
boxplot(V(label==1,gra),'Positions',1, 'Colors',[0.00,0.45,0.74], 'Symbol','+', 'Widths',0.5)
hold on
boxplot(V(label==5,gra),'Positions',2, 'Colors',[0.47,0.67,0.19], 'Symbol','+', 'Widths',0.5)
boxplot(V(label==3,gra),'Positions',3, 'Colors',[1.00,0.00,0.00], 'Symbol','+', 'Widths',0.5)
boxplot(V(label==4,gra),'Positions',4, 'Colors',[0.49,0.18,0.56], 'Symbol','+', 'Widths',0.5)
xlim([0.5,4.5])
ylim([-0.5,0.7])
xticks([1,2,3,4])
xticklabels({'ASD','ADHD','VCI','MCI/AD'})
ylabel(['Gradient ' num2str(gra)])

hold on;
DATA.d1=V(label==1,gra);
DATA.d2=V(label==5,gra);
DATA.d3=V(label==3,gra);
DATA.d4=V(label==4,gra);
sigstar({[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]},[ranksum(DATA.d1,DATA.d2),ranksum(DATA.d1,DATA.d3),ranksum(DATA.d1,DATA.d4),...
   ranksum(DATA.d2,DATA.d3),ranksum(DATA.d2,DATA.d4),ranksum(DATA.d3,DATA.d4)]);



% outliers in ASD
[V,lambda] = diffusion_mapping(disM(:,:,3), 3, 0.5, 0);

low1=prctile(V(label==5,1),10);
up2=prctile(V(label==5,2),90);

pos = (label==1)&(V(:,2)<=up2)&(V(:,1)>low1);
%pos2 = find((label==1)&(V(:,2)>0.0));
pos2 = find((label==1)&~pos);
pos = find((label==1)&(V(:,2)<=up2)&(V(:,1)>low1));

% V = cmdscale(disM(:,:,3));
scatter(V(pos2,1),V(pos2,2),'filled');

hold on
scatter(V(pos,1),V(pos,2),'filled');
hold on
scatter(V(label==5,1),V(label==5,2),'filled');


ll=length(pos2);
subinfo=subinforevised(pos2,:);

subname=subinfo(:,1);
pos = zeros(1,1112);
for i=1:1112
    for j=1:ll
        if strcmp(PhenotypicV10b{i,3},subname{j})
            pos(i)=1;
        end
    end
end
out=PhenotypicV10b(find(pos),:);

pos = find((label==1)&(V(:,2)<=up2)&(V(:,1)>low1));

ll=length(pos);
subinfo=subinforevised(pos,:);

subname=subinfo(:,1);
pos = zeros(1,1112);
for i=1:1112
    for j=1:ll
        if strcmp(PhenotypicV10b{i,3},subname{j})
            pos(i)=1;
        end
    end
end
major = PhenotypicV10b(find(pos),:);

pp1=[];
pp2=[];
for col=[6,10:12,16:39,44:75]
    data1 = zeros(1,size(out,1));
    data2 = zeros(1,size(major,1));
    for i=1:size(out,1)
        data1(i)=out{i,col};
    end
    for i=1:size(major,1)
        data2(i)=major{i,col};
    end
    data1(data1==-9999)=[];
    data2(data2==-9999)=[];

    [p]=ranksum(data1,data2);
    pp1=[pp1,p];
    [~,p]=ttest2(data1,data2);
    pp2=[pp2,p];
end

sum(data1==2)

propotion=[138/207,(48+17+4)/207;174/227,(35+16+2)/227];

bar(propotion)
ylabel('Propotion (%)')
xticks([1,2])
xticklabels({'Outlier','Major'})
sigstar([1,2],[0.0208]);

data1 = zeros(1,size(out,1));
data2 = zeros(1,size(major,1));
for i=1:size(out,1)
    data1(i)=out{i,5};
end
for i=1:size(major,1)
    data2(i)=major{i,5};
end
data1(data1==-9999)=[];
data1(isnan(data1))=[];
data2(data2==-9999)=[];
data2(isnan(data2))=[];

for i=1:4
    sum(data2==i)
end

figure;
% ViolinPlot(data2,'Positions',1,'Colors','b')
% hold on
% boxplot(data1,'Positions',2,'PlotStyle','compact', 'Colors','r')
DATA.d1=data2;
DATA.d2=data1;
vio=violinplot(DATA);
vio(1).ViolinColor=[0,0,1];
vio(2).ViolinColor=[1,0,0];

xlim([0.5,2.5])
ylim([0,35])
xticks([1,2])
xticklabels({'Major','Outlier'})


% correlation to age and gender
gra=1;
boxplot(V(gender==1,gra),'Positions',1,'Colors','b')
hold on
boxplot(V(gender==0,gra),'Positions',2,'Colors','r')
xlim([0.5,2.5])
ylim([-0.8,0.8])
xticks([1,2])
xticklabels({'M','F'})
ylabel(['Gradient' num2str(gra)])
% ranksum(V(gender==1,gra),V(gender==0,gra))
sigstar({[1,2]},[ranksum(V(gender==1,gra),V(gender==0,gra))]);

gra=1;
scatter(age,V(:,gra))
[r,p]=corr(age,V(:,gra))
xlabel('Age (yrs)')
ylabel(['Gradient' num2str(gra)])

intp=ones(length(age),1);
[b,~,tempy] = regress(V(:,gra),[age,intp]);
yf=[age,intp]*b;
hold on;
plot(age,yf,'k--')