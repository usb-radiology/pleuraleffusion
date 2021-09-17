function Lung_par_cor(nodeinfoPath)

run /opt/nora/src/matlab/DPX_startup.m
% Range=eval(Range);

DPX_Project('ThoraxCTD')
addpath(genpath('/data/Projects/OpQuant/LungsSegmentation'))
addpath(genpath('/data/Projects/Pleuearerguss/'))
% SUB=DPX_SQL_query(['select * from studies where (Stag LIKE "%/%covid%/%") '])

% addpath(genpath('/mnt/nfs_data/studies/shan/Projects/lungstage/LungsSegmentation'))
% addpath('/mnt/nfs_data/studies/shan/Projects/lungstage')

% Range(Range>length(SUB))=[];

%%
%%
nodeinfo=DPX_readJSON(nodeinfoPath)


psid=[nodeinfo.PatientID,'#',nodeinfo.StudyID]
label_input=DPX_selectFiles([psid, ' downsampling/pred_lungpleu_final.nii.gz'])


P=load_untouch_nii(label_input{1});

raw_input=DPX_selectFiles([psid, ' downsampling/raw.nii'])

Praw=load_untouch_nii(raw_input{1});


Praw.img=trans_image(Praw.img,Praw,0,'linear');
P.img=trans_image(P.img,P,0);


maskLung  = ph4dLM_labelAndCleanLungsMask(uint8(P.img>0), true);

masktype=class(maskLung);
display('stopped after here')


se = strel('sphere',1);

x0=0;
y0=0;
% maskLung_=permute(maskLung,[1 3 2]);
dim = size(maskLung);

mask_mid = (maskLung~=0);
% mask_mid = (maskLung~=0);
mask_side = mask_mid;
mask_connect = mask_mid;
mask_abdome = mask_mid;
% zidx=find(sum(maskLung,[1,2])~=0);

for z=1:dim(3)
    im=maskLung(:,:,z);
    im=im.*cast(find_max_component(im~=0,2),class(im));
    display(sprintf('process slice %d',z))
    iml=(im==1);
    imr=(im==2);
    
    if nnz(iml)~=0 & nnz(imr)~=0
        Y_L=find(sum(iml,1));
        Y_L_top=max(Y_L);
        Y_L_bottom=min(Y_L);
        [pks,X_L_top] = findpeaks(double(iml(:,Y_L_top)));
        [pks,X_L_bottom] = findpeaks(double(iml(:,Y_L_bottom)));
        
        
        Y_R=find(sum(imr,1));
        Y_R_top=max(Y_R);
        Y_R_bottom=min(Y_R);
        [pks,X_R_top] = findpeaks(double(imr(:,Y_R_top)));
        [pks,X_R_bottom] = findpeaks(double(imr(:,Y_R_bottom)));
        
        newim=cast(zeros(size(im)),class(im));
        newim=connect_border_point(Y_L(end),X_R_top,Y_R_top,iml);
        newim=newim+connect_border_point(Y_R(end),X_L_top,Y_L_top,imr);
        if abs(Y_R_bottom-Y_L_bottom)<dim(2)*0.4;
            
            newim=newim+connect_border_point(Y_L(1),X_R_bottom,Y_R_bottom,iml);
            newim=newim+connect_border_point(Y_R(1),X_L_bottom,Y_L_bottom,imr);
        end
        
        mask_connect(:,:,z)= mask_connect(:,:,z)+imdilate(newim~=0,se);
        mask_connect(:,:,z)=imfill(mask_connect(:,:,z)>0,'holes');
        
        
    end
    
    
    for y=1:dim(2)
        Y_line = squeeze(im(:,y));
        
        if length(unique(Y_line))>=3
            
            CC = bwconncomp(Y_line==0);
            
            if CC.NumObjects>=3
                S = regionprops(CC,'Centroid','Area');
                Cent = reshape([S.Centroid],[2 length(S)]);
                Cent = Cent(1,:).*Cent(2,:);
                %                 idx=[S.Area]>;
                [~,idx_area]=sort([S.Area]);
                mid_dis=abs(Cent-dim(2)/2);
                [~,idx_mid]=sort(mid_dis);
                
                %                 if CC.NumObjects==3
                %                                     %
                %                     f_area=[S.Area]>=round(dim(2)*0.01) & [S.Area]<round(dim(2)*0.45);
                %                 else
                %                                 f_area=[S.Area]>=round(dim(2)*0.05);
                f_area=[S.Area]>=round(dim(2)*0.01) & [S.Area]<round(dim(2)*0.45);
                
                
                idx_mid_=idx_mid(f_area);
                mid_dis_=mid_dis(f_area);
                f_mid=mid_dis < round(dim(2)*0.2);
                
                idx=f_area & f_mid;
                
                
                if length(idx_mid)>=3 & nnz(idx)>0;
                    [mid_min,idx_mid_]=sort(mid_dis(idx));
                    idx_=find(mid_dis==mid_min(1));
                    mask_mid(CC.PixelIdxList{idx_(1)},y,z)=1;
                    
                end
                
                
                side_idx=[CC.PixelIdxList{idx_mid(end)}' CC.PixelIdxList{idx_mid(end-1)}'];
                mask_side(side_idx,y,z)=1;
                
            end
        end
    end
    
end

% mask_mid_0=mask_mid;

% mask_mid=mask_connect+mask_mid;
mask_mid=cast(mask_mid,masktype);
mask_mid(maskLung~=0)=0;
% mask_mid(mask_side)=0;
mask_side(maskLung~=0)=0;
% mask_mid(mask_connect~=0)=0;
mask_side(mask_connect~=0)=0;

mask_side=find_max_component(mask_side,2);

% maskLung=permute(maskLung,[1 3 2]);
% mask_mid=permute(mask_mid,[1 3 2]);

se = strel('sphere',4);
mask_side=imerode(mask_side,se);
% % mask_abdome=find_max_component(mask_abdome,2);
mask_side=imdilate(mask_side,se);

mask_side0=mask_side;
mask_mid0=mask_mid;



%% correct z-achse.


maskLung_=permute(maskLung,[1 3 2]);
mask_mid_=permute(mask_mid,[1,3,2]);
mask_abdome=zeros(size(maskLung_),class(maskLung_));
dim=size(maskLung_);

for z=1:size(maskLung_,3)
    display(sprintf('process slice %d',z))
    im0=cast(zeros(dim(1:2)),class(mask_abdome));
    
    for l=1:2
        
        Xmin=[];
        Xmin=min(find(sum(maskLung_(:,:,z)==l,1)));
        
        if ~(isempty(Xmin));
            im_=im0;
            
            im=(maskLung_(:,1:Xmin+30,z)==l);
            
            YN=[];
            YN=sum(im~=0,2);
            YN=find(YN>0);
            
            
            if ~(isempty(YN));
                for x=1:length(YN)
                    Y=find(im(YN(x),:)~=0);
                    
                    %                     Ymin=Y(1)-25;
                    %                     if Ymin<1
                    Ymin=1;
                    %                     end
                    
                    im_(YN(x),Ymin:Y(1))=1;
                    %                     mask_abdome_vertical()
                    %
                end
                
                mask_abdome(:,:,z)=mask_abdome(:,:,z)+find_max_component(im_);
                
            end
        end
        
    end
    
    
    for x=1:dim(1)
        
        X=squeeze(mask_abdome(x,:,z) + mask_mid_(x,:,z)*2);
        
        mid_idx=find(X==2);
        abdome_idx=find(X==1);
        
        if ~isempty(mid_idx) & isempty(abdome_idx)
            mask_abdome(x,1:mid_idx(1),z)=1;
            %             mask_mid_(x,mid_idx(1):
        elseif ~isempty(mid_idx) & ~isempty(abdome_idx)
            
            mask_mid_(x,mid_idx(1):abdome_idx(1),z)=1;
            
            
        end
        
        
    end
    
    im=mask_abdome(:,:,z)+mask_mid_(:,:,z);
    im_=imfill(im,'holes');
    mask_abdome(:,:,z)= mask_abdome(:,:,z) + (im-im_);
    
end

mask_mid_=permute(mask_mid_,[1 3 2]);
mask_abdome=permute(mask_abdome,[1 3 2]);
mask_abdome=imerode(mask_abdome,se);
mask_abdome=find_max_component(mask_abdome,3);
mask_abdome=imdilate(mask_abdome,se);


se = strel('sphere',4);
mask_abdome=imerode(imdilate(mask_abdome,se),se);


% mask_abdome

% mask_abdome(mask_mid)=0;
% mask_abdome(mask_side)=0;
% mask_abdome=(mask_mid & mask_abdome);

%%
Ptmp=P;

dim=size(mask_mid);
% remove bone region
se = strel('sphere',2);
mask_mid_=mask_mid;
% mask_mid_=imdilate((mask_mid+mask_connect)>0,se);
% mask_mid_(mask_abdome~=0)=0;
int_off=Praw.hdr.dime.scl_inter;
mask_bone=(Praw.img>(120-int_off));

mask_bone=imdilate(mask_bone,se);
mask_bone=imerode(mask_bone,se);
% mask_bone=find_max_component(mask_bone,10);

% mask_mid_(mask_bone_~=0)=0;

% % select largest region
mask_mid_(P.img~=0)=0;
mask_mid_(mask_bone~=0)=0;


se = strel('sphere',6);

mask_mid_=imerode(mask_mid_,se);
mask_mid_=find_max_component(mask_mid_);
mask_mid_=imdilate(mask_mid_,se);

se = strel('sphere',6);

mask_mid_=((cast(P.img,class(mask_mid_))+mask_mid_)~=0);
mask_mid_=imdilate(mask_mid_,se);
mask_mid_=imfill(mask_mid_,'holes');
mask_mid_=imerode(mask_mid_,se);
mask_mid_(P.img~=0)=0;

se = strel('sphere',1);

mask_mid_=imerode(mask_mid_,se);
mask_mid_=find_max_component(mask_mid_);
mask_mid_=imdilate(mask_mid_,se);

% exclude CAD
mask_mid_ = imfill(mask_mid_,'holes');
mask_bone (mask_mid_~=0)=0;


mask_lung_path=DPX_getOutputLocation([psid, ' ph4d/mask_Lung.nii.gz'])
mask_mid_path=DPX_getOutputLocation([psid, ' ph4d/mask_mid.nii.gz'])
mask_side_path=DPX_getOutputLocation([psid, ' ph4d/mask_side.nii.gz'])
mask_bone_path=DPX_getOutputLocation([psid, ' ph4d/mask_bone.nii.gz'])
mask_abdome_path=DPX_getOutputLocation([psid, ' ph4d/mask_abdome.nii.gz'])

% fill lung,if is mid
mask_mid_=permute(mask_mid_,[1 3 2]);
dim_=size(mask_mid_);
for z=1:dim_(3)
    lung_fill = imfill(maskLung_(:,:,z),'holes');
    %    mask_tmp = maskLung_(:,:,z);
    mask_tmp = (lung_fill & mask_mid_(:,:,z));
    %     mask_tmp = lung_fill(mask_tmp);
    maskLung_(:,:,z)= cast(mask_tmp,masktype) + maskLung_(:,:,z);
    
end

mask_mid_(maskLung_~=0)=0;
mask_mid_=permute(mask_mid_,[1 3 2]);


maskLung_=permute(maskLung_,[1 3 2]);

Ptmp.img = trans_image(mask_mid_,Ptmp,1);
Ptmp.hdr.dime.scl_inter=0;
Ptmp.hdr.dime.glmax=2;
save_untouch_nii(Ptmp,mask_mid_path);



Ptmp.img = trans_image(maskLung_,Ptmp,1);
Ptmp.hdr.dime.scl_inter=0;
Ptmp.hdr.dime.glmax=2;
save_untouch_nii(Ptmp,mask_lung_path);



% se = strel('sphere',10);
% mask_bone_=imdilate(mask_bone,se);


% fill the holes in bone:
mask_bone=permute(mask_bone,[1,3,2]);
dim_=size(mask_bone);
for z=1:dim_(3)
    mask_bone(:,:,z)=imfill(mask_bone(:,:,z),'holes');
end
mask_bone=permute(mask_bone,[1,3,2]);
mask_bone=remove_small_lesions(mask_bone,10);
Ptmp.img = trans_image(mask_bone,Ptmp,1);
save_untouch_nii(Ptmp,mask_bone_path);

%%
mask_side(mask_abdome~=0)=0;

mask_side_=mask_side;
mask_side_(mask_bone)=0;


mask_abdome_=mask_abdome;


% mask_abdome(mask_mid_~=0)=0
mask_abdome_(mask_bone~=0)=0;
mask_abdome_(maskLung~=0)=0;
mask_abdome_=find_max_component(mask_abdome,2);

mask_abdome_=find_max_component_2d(mask_abdome,[1 2 3],2);
mask_abdome_=find_max_component_2d(mask_abdome,[2 3 1],1);
mask_abdome_=permute(mask_abdome_,[3 1 2]);

% mask_abdome_(rmim~=0)=1;
mask_abdome_(mask_bone~=0)=0;
mmask_abdome_(maskLung_~=0)=0;
mask_abdome_(mask_mid_~=0)=0;

Ptmp.img = trans_image(mask_abdome_,Ptmp,1);
save_untouch_nii(Ptmp,mask_abdome_path);

se = strel('sphere',3);

maskLung_=cast(imdilate(maskLung~=0,se),masktype);
mask_side_= cast(mask_side_,masktype) + (maskLung_ - maskLung);
mask_side_(mask_bone~=0)=0;
mask_side_(mask_abdome_~=0)=0;
mask_side_(mask_mid~=0)=0;
mask_side_=find_max_component(mask_side_,2);
Ptmp.img = trans_image(mask_side_,Ptmp,1);
save_untouch_nii(Ptmp,mask_side_path);


%% create parcellation for lunge boarder
mask_side_inside_path=DPX_getOutputLocation([psid, ' lung_mask/mask_side.nii.gz'])


 CC = bwconncomp(maskLung==1);
    
S = regionprops3(CC,'BoundingBox','Centroid','Volume');
[m i] =max(S.Volume);
S=S(i,:);
    % LungWitdh=length(find(sum(squeeze(Mask(:,round(S.Centroid(2)),:)),2)));
    
se = strel('sphere',floor(S.BoundingBox(4)/12));

%     mask_side_inside=cast((imdilate((mask_side0~=0 | mask_abdome_~=0),se) & maskLung~=0),masktype);
mask_side_inside=remove_small_lesions((maskLung==0 & mask_mid0==0),5000);

mask_side_inside=cast((imdilate(mask_side_inside,se) & maskLung~=0),masktype);





Ptmp.img = trans_image(mask_side_inside,Ptmp,1);

save_untouch_nii(Ptmp,mask_side_inside_path);

% %% create parcellation for lunge front and back
% 
% 
% % idx_z_l=squeeze(sum(maskLung==1,[1 2]));
% 
% mask_top=maskLung;
% mask_mid_horz=maskLung;
% mask_bottom=maskLung;
% mask_front=maskLung;
% mask_back=maskLung;
% 
% 
% for s=1:2
%     
%     halfLung=(maskLung==s);
%     [halfLung,rim]=find_max_component(halfLung);
%     CC = bwconncomp(halfLung);
%     S = regionprops(CC,'Centroid','BoundingBox');
%     center=round(S.Centroid);
%     
%     
%     halfLung_=halfLung;
%     halfLung_(:,1:center(1),:)=0;
%     mask_back(halfLung_)=0;
%     halfLung_=halfLung;
%     halfLung_(:,center(1)+1:end,:)=0;
%     mask_front(halfLung_)=0;
%     
%     %         idxx=find(squeeze(sum(maskLung==s,[2 3]))>0);
%     %
%     %         idxy=find(squeeze(sum(maskLung==s,[1 3]))>0);
%     %
%     %         idxz=find(squeeze(sum(maskLung==s,[1 2]))>0);
%     
%     center=[round(S.BoundingBox(4)/3) round(S.BoundingBox(5)/3) round(S.BoundingBox(6)/3)];
%     
%     %         center(2)=dim(2)-center(2);
%     
%     %         center=[center(2) center(1) center(3)];
%     
%     halfLung_=halfLung;
%     halfLung_(:,:,S.BoundingBox(3)+1:S.BoundingBox(3)+center(3))=0;
%     mask_bottom(halfLung_) = 0;
%     
%     
%     halfLung_=halfLung;
%     halfLung_(:,:,S.BoundingBox(3)+center(3)+1:S.BoundingBox(3)+2*center(3))=0;
%     mask_mid_horz(halfLung_) = 0;
%     
%     halfLung_=halfLung;
%     halfLung_(:,:,S.BoundingBox(3)+2*center(3)+1:end)=0;
%     mask_top(halfLung_)=0;
%     
%     
%     
% end
% %
% %     Ptmp.img = trans_image(maskLung,Ptmp,1);
% %     save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_maskLung.nii.gz']));
% 
% Ptmp.img = trans_image(mask_bottom,Ptmp,1);
% save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_bottom.nii.gz']));
% Ptmp.img = trans_image(mask_mid_horz,Ptmp,1);
% save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_mid_horz.nii.gz']));
% Ptmp.img = trans_image(mask_top,Ptmp,1);
% save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_top.nii.gz']));
% Ptmp.img = trans_image(mask_front,Ptmp,1);
% save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_front.nii.gz']));
% Ptmp.img = trans_image(mask_back,Ptmp,1);
% save_untouch_nii(Ptmp,DPX_getOutputLocation([psid, ' lung_mask/mask_back.nii.gz']));



[dpath,fname,~]=fileparts(mask_mid_path);
DPX_addFiles([dpath,' TAG(mask)'])

[dpath,fname,~]=fileparts(mask_side_inside_path);
DPX_addFiles([dpath,' TAG(mask)'])


display('done')
DPX_SQL_update_tag(psid,'STAG(parcellation)')

end


function newim=find_max_component_2d(im,permuteMatrix,num_maxarea)

im=permute(im,permuteMatrix);
dim=size(im);
newim=cast(zeros(dim),class(im));

for z=1:dim(3)
    newim(:,:,z)=find_max_component(im(:,:,z),num_maxarea);
    
end


end


function [newim rmim]=find_max_component(im,num_maxarea)

if nargin<2
    num_maxarea=1;
end


dim=size(im);
newim=cast(zeros(dim),class(im));
rmim=im;

CC = bwconncomp(im>0);

if CC.NumObjects==0
    newim=im;
else
    
    S = regionprops(CC,'Centroid','Area');
    newim=cast(zeros(dim),class(im));
    [~,idx_max]=sort([S.Area],'descend');
    
    if CC.NumObjects<num_maxarea
        num_maxarea=CC.NumObjects;
    end
    
    
    for n=1:num_maxarea;
        newim(CC.PixelIdxList{idx_max(n)})=1;
    end
    
    
end
rmim(newim~=0)=0;


end





function newim =remove_small_lesions(im,vs)


dim=size(im);
newim=cast(zeros(dim),class(im));
rmim=im;

CC = bwconncomp(im>0);

if CC.NumObjects==0
    newim=im;
else
    
    S = regionprops(CC,'Centroid','Area');
    newim=cast(zeros(dim),class(im));
    [~,idx_max]=sort([S.Area],'descend');
    
    sidx = find([S.Area]>vs);
    
    for n=1:nnz(sidx);
        newim(CC.PixelIdxList{sidx(n)})=1;
    end
    
    
end


end

function img=trans_image(img,P,inverse,interp);
dim=size(img);
if nargin<4
    interp='nearest';
end
t= spm_imatrix(get_mat(P));
t(1:3)=[0 0 0];

t(7:9)=sign(t(7:9));

if ~inverse
    t= spm_matrix(t);
else
    t=inv(spm_matrix(t));
end

tform = affine3d(t);

img_=imwarp(img,tform,interp);

img=img_(1:dim(1),1:dim(2),1:dim(3));



end



% getData: extract data
% function data = getData(vol,T)
function data = getData(data,mat,interp)

dim=size(data);
[nx,ny,nz] = ndgrid((1:dim(1)),...
    (1:dim(2)),...
    (1:dim(3)));
T=cat(4,nx,ny,nz);

data = tformarray(data , ...
    maketform('affine',mat'), ...
    makeresampler({interp interp interp}, 'fill'),...
    [1 2 3], [1 2 3], [], T, 0);
end



function mat=get_mat(rl)
mat=eye(4,4);
mat(1,:)=rl.hdr.hist.srow_x(1:4);
mat(2,:)=rl.hdr.hist.srow_y(1:4);
mat(3,:)=rl.hdr.hist.srow_z(1:4);
end

