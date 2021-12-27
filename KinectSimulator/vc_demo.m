clear all
close all

% Add opcodemesh folders
folder_path = pwd;
path_opcodemesh = [folder_path '/opcodemesh'];
if ~contains(path,path_opcodemesh)
    disp('Adding opcodemesh path...')
    addpath(genpath(path_opcodemesh))
end

%% DEMO TEAPOT ============================================================
load('example_CAD/Teapot.mat')

% Shift CAD model to further depth
vertex_teapot = vertex;
vertex_teapot(3,:) = vertex_teapot(3,:) + 600;
face_teapot = face;
norm_teapot = normalf;

[a,b,c] = KinectSimulator_IR(vertex_teapot,face_teapot,norm_teapot,...
    'default','default','default','max','imgrng',[400 1000],...
    'window',[7 9],'refine',10,'displayIR','off');

figure, imshow(a)
figure, imshow(b)
figure, imshow(c)


DpthImg = KinectSimulator_Depth(vertex_teapot,face_teapot,norm_teapot,...
    'default','default','default','max','imgrng',[400 1000],...
    'window',[9 9],'refine',10,'displayIR','off');
figure, imshow(DpthImg,[])
title('Noisy depth image of teapot in front of flat wall')
drawnow



return
