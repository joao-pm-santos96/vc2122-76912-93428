clear all
close all

% Add opcodemesh folders
folder_path = pwd;
path_opcodemesh = [folder_path '/opcodemesh'];
if ~contains(path,path_opcodemesh)
    disp('Adding opcodemesh path...')
    addpath(genpath(path_opcodemesh))
end

%% DEMO BOX GRID ==========================================================
load('example_CAD/BoxGrid.mat')

box_width = 96;  % mm
box_depth = 54;  % mm
wallDist  = 854; % mm

% Scale size of box and shift grid to further depth
vertex_grid = vertex;
vertex_grid(1:2,:) = vertex_grid(1:2,:).*box_width;
vertex_grid(3,:) = vertex_grid(3,:).*box_depth + wallDist-box_depth;
face_grid = face;
norm_grid = normalf;
GridRng = [min(vertex_grid(3,:)) max(vertex_grid(3,:))];

[a,b,c] = KinectSimulator_IR(vertex_grid,face_grid,norm_grid); 

close all
figure, imshow(a)
figure, imshow(b)
figure, imshow(c)


return


DpthImg = KinectSimulator_Depth(vertex_xbox,face_xbox,norm_xbox)%,...
    %'simple','default','default',[],'imgrng',[200 1000],'subray',[5 9]);




figure, imshow(DpthImg,[])
title('Noisy depth image of xbox controller in front of nothing')
drawnow
