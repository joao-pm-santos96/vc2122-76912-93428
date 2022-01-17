clear all
close all

% Add opcodemesh folders
folder_path = pwd;
path_opcodemesh = [folder_path '/opcodemesh'];
if ~contains(path,path_opcodemesh)
    disp('Adding opcodemesh path...\n')
    addpath(genpath(path_opcodemesh))
end

load('example_CAD/Teapot.mat')

% Shift CAD model to further depth
vertex_teapot = vertex;
vertex_teapot(3,:) = vertex_teapot(3,:) + 600;
face_teapot = face;
norm_teapot = normalf;

DpthImg = KinectSimulator_Depth(vertex_teapot,face_teapot,norm_teapot,...
    'default','default','default','max','imgrng',[400 1000]);

figure, imshow(DpthImg,[])
title('Noisy depth image of teapot in front of flat wall')
drawnow
