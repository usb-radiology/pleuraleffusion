
function splitte_lung(input, output)

[sdir,sname,~]  = fileparts(mfilename('fullpath'));

[rep_dir,modpath,~]=fileparts(sdir);


addpath(genpath([rep_dir,'/third']))

P=load_untouch_nii(input);


img = trans_image(P.img,P,0,'nearest');

img  = ph4dLM_labelAndCleanLungsMask(uint8(img>0), true);

P.img = trans_image(img,P,1,'nearest');


save_untouch_nii(P,output)


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

