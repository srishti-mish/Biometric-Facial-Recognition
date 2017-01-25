clear;
%img_files = dir('/Users/srishti/Documents/MATLAB/GallerySet/*.pgm');
for i=1:100
    img_files(i).name=strcat('subject',int2str(i),'_img1.pgm');
end
img_vectors = zeros(2500,length(img_files));

%creating face vectors
for i = 1 : length(img_files)
    filename = strcat('/Users/srishti/Documents/MATLAB/GallerySet/',img_files(i).name);
    I = imread(filename);
    img_vectors(:,i) = I(:);
    %figure, imshow(I);
end

mean_matrix = zeros(size(img_vectors,2),1);

%normalising the face vectors
for i=1:size(img_vectors,1)
    sum=0;
    for j=1:size(img_vectors,2)
        sum = sum + img_vectors(i,j);
    end
    mean_matrix(i)=sum/j;
end
for i=1:size(img_vectors,1)
    for j=1:size(img_vectors,2)
        img_vectors(i,j)= img_vectors(i,j)-mean_matrix(i);
    end
end
%covariance matrix
cov = zeros(size(img_vectors,2),size(img_vectors,2));
cov = img_vectors'*img_vectors;

[eig_vec,eig_val] = eig(cov);
Eig_vec = img_vectors * eig_vec;

%weight_vector = zeros(2500,size(img_vectors,2));
weight_vector = Eig_vec'*img_vectors;

%mapping 1st three eigen faces to original dimension
pca=zeros(size(img_vectors,1),3);

%question 1
for i=1:3
    vec = Eig_vec(:,101-i);
    vec=vec+mean_matrix;
    for j=1:2500
        %floor((j-1)/50+1)
        mat(i,mod(j,50)+1,floor((j-1)/50+1)) = uint8(vec(j));
    end
end    
% temp(:,:)=mat(1,:,:);
% imshow(temp);    
% temp(:,:)=mat(2,:,:);
% imshow(temp);    
% temp(:,:)=mat(3,:,:);
% imshow(temp);

%finding weight vectors

%adding back the average face

%Probeset
%probe_img_files = dir('/Users/srishti/Documents/MATLAB/ProbeSet/*.pgm');
s=1;
for i=1:100
    probe_img_files(s).name=strcat('subject',int2str(i),'_img2.pgm');
    probe_img_files(s+1).name=strcat('subject',int2str(i),'_img3.pgm');
    s=s+2;
end
probe_img_vectors = zeros(2500,length(probe_img_files));
for i = 1 : length(probe_img_files)
    filename = strcat('/Users/srishti/Documents/MATLAB/ProbeSet/',probe_img_files(i).name);
    I = imread(filename);
    probe_img_vectors(:,i) = I(:);
    %figure, imshow(I); 
end

for i=1:size(probe_img_vectors,1)
    for j=1:size(probe_img_vectors,2)
        probe_img_vectors(i,j)= probe_img_vectors(i,j)-mean_matrix(i);
    end
end

%covariance matrix
%probe_cov = zeros(size(probe_img_vectors,2),size(probe_img_vectors,2));
%probe_cov = probe_img_vectors'*probe_img_vectors;
% [probe_eig_vec,probe_eig_val] = eig(probe_cov);
c=1;
for nEig=10:10:100
    for nImg=1:size(probe_img_vectors,2)
        cWeiVec=Eig_vec(:,101-nEig:100)'*probe_img_vectors(:,nImg);
        cMinDis=inf;
        for i=1:size(weight_vector,2)
            dis=norm(weight_vector(101-nEig:100,i)-cWeiVec);
            if(dis<cMinDis)
                cMinDis=dis;
                cRecog=i;
            end
        end
        recogMat(c,nImg)=cRecog;
    end
    c=c+1;
end
c=1;
for nImg=1:size(probe_img_vectors,2)
    cEucMinDis=inf;
    for i=1:size(weight_vector,2)
        eucDis=norm(probe_img_vectors(:,nImg)-img_vectors(:,i));
        if(eucDis<cEucMinDis)
            cEucMinDis=eucDis;
            cEucRecog=i;
        end
    end
    recogEucMat(c,nImg)=cEucRecog;
end
PCARecogPerf=zeros(10,1);
EucRecogPerf=zeros(1);
for d=1:10
    s=1;
    for i=1:100
        if(recogMat(d,s)==i)
            PCARecogPerf(d)=PCARecogPerf(d)+1;
        end
        if(recogMat(d,s+1)==i)
            PCARecogPerf(d)=PCARecogPerf(d)+1;
        end
        s=s+2;
    end
end
s=1;
for i=1:100
    if(recogEucMat(1,s)==i)
        EucRecogPerf(1)=EucRecogPerf(1)+1;
    end
    if(recogEucMat(1,s+1)==i)
        EucRecogPerf(1)=EucRecogPerf(1)+1;
    end
    s=s+2;
end

PCARecogPerf(:)=PCARecogPerf/2;
EucRecogPerf(:)=EucRecogPerf/2;
genderMat=tdfread('Gender.txt',',');
c=1;
c1=1;
clusCen=zeros(20,100);
clusId=zeros(10,2);
for nEig=10:10:100
    distM=squareform(pdist(weight_vector(101-nEig:100,:)'));
    [kmat(c,:) clusCen(c1:c1+1,1:nEig)]=kmeans(weight_vector(101-nEig:100,:)',2);
    disp(sprintf('Dunns index for kmeans %d', dunns(2,distM,kmat(c,:))));
    numCorr1=0;
    numCorr2=0;
    for i=1:100
        if((strcmp(genderMat.gen(i,:),'male')==0&&kmat(c,i)==1)||(strcmp(genderMat.gen(i,:),'female')==0&&kmat(c,i)==2))
            numCorr1=numCorr1+1;
        end
        if((strcmp(genderMat.gen(i,:),'male')==0&&kmat(c,i)==2)||(strcmp(genderMat.gen(i,:),'female')==0&&kmat(c,i)==1))
            numCorr2=numCorr2+1;
        end        
    end
    if(numCorr1>numCorr2)
        recogPercent(c)=numCorr1;
        for i=1:size(genderMat.gen,1)
            if(strcmp(genderMat.gen(i,:),'male')==0)
                ids(i)=1;
            else
                ids(i)=2;
            end
        end
        clusId(c,1)=1;
        clusId(c,2)=2;
        disp(sprintf('NMI index for kmeans %f', nmi(kmat(c,:),ids)));
    else
        recogPercent(c)=numCorr2;
        for i=1:size(genderMat.gen,1)
            if(strcmp(genderMat.gen(i,:),'male')==0)
                ids(i)=2;
            else
                ids(i)=1;
            end
        end
        clusId(c,1)=2;
        clusId(c,2)=1;
        disp(sprintf('NMI index for kmeans %f', nmi(kmat(c,:),ids)));
    end
    c=c+1;
    c1=c1+2;
end
myClus=zeros(10,200);
c=1;
cWeiVec=zeros(200,100);
for nEig=10:10:100
    for nImg=1:200
        cWeiVec(nImg,1:nEig)=Eig_vec(:,101-nEig:100)'*probe_img_vectors(:,nImg);
        disTemp1=norm(cWeiVec(nImg,1:nEig)-clusCen(2*(c-1)+1,1:nEig));
        disTemp2=norm(cWeiVec(nImg,1:nEig)-clusCen(2*(c-1)+2,1:nEig));
        if(disTemp1<disTemp2)
            myClus(c,nImg)=clusId(c,1);
        else
            myClus(c,nImg)=clusId(c,2);
        end
    end
    for i=1:200
        if(strcmp(genderMat.gen(i),'male')==0)
            targetGen(c,i)=1;
        else
            targetGen(c,i)=2;
        end
    end
    distM=squareform(pdist(cWeiVec));
    disp(sprintf('1 Dunns index for kmeans %d', dunns(2,distM,myClus(c,:))));
    disp(sprintf('1 NMI index for kmeans %f', nmi(myClus(c,:),targetGen(c,:))));
    c=c+1;
end