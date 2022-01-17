clear all
close all

% Add opcodemesh folders
folder_path = pwd;
path_opcodemesh = [folder_path '/opcodemesh'];
if isempty(strfind(path,path_opcodemesh))
    disp('Adding opcodemesh path...\n')
    addpath(genpath(path_opcodemesh))
end

%% DEMO FLAT WALLS BETWEEN DEFAULT MIN AND MAX DEPTHS =====================
if false
  testMinDepth = 1000; % mm
  testMaxDepth = 3000; % mm

  % Set parameters of wall CAD model
  vertex_wall = [2e4  2e4 -2e4 -2e4;...
                 7e3 -7e3  7e3 -7e3;...
                 0    0    0    0];
  face_wall   = [1 1;...
                 2 3;...
                 4 4];
  norm_wall   = [0  0;...
                 0  0;...
                -1 -1];

  vertex_wall(3,:) = testMinDepth*ones(1,4);
  DpthImg = vc_KinectSimulator_Depth(vertex_wall,face_wall,norm_wall);
  figure, imshow(DpthImg,[])
  title('Noisy depth image of flat wall at mm')
  drawnow
end

%% DEMO CORNER OF ROOM WITH TILTED WALL ===================================
if false
  tilt_Angle  = 60;   % deg
  tilt_Center = 1000; % mm
  tilt_X = -tilt_Center/tand(tilt_Angle);
  tilt_Z =  tilt_Center + 1e5*tand(tilt_Angle);

  % Set parameters of tilted wall CAD model
  vertex_tilt = [1e5     1e5     tilt_X  tilt_X;...
                 1e5    -1e5     1e5    -1e5;...
                 tilt_Z  tilt_Z  0       0];
  face_tilt   = [1 1;...
                 2 3;...
                 4 4];
  norm_tilt   = [sind(tilt_Angle)  sind(tilt_Angle);...
                 0                 0;...
                -cosd(tilt_Angle) -cosd(tilt_Angle)];

  % Add floor to CAD model
  vertex_floor = [1e5     1e5  -1e5     -1e5;...
                 -5e2    -5e2  -5e2     -5e2;...
                  tilt_Z  0     tilt_Z   0];
  vertex_tilt  = [vertex_tilt vertex_floor];

  face_floor = [size(vertex_tilt,2)-3 size(vertex_tilt,2)-3;...
                size(vertex_tilt,2)-2 size(vertex_tilt,2)-1;...
                size(vertex_tilt,2)   size(vertex_tilt,2)];
  face_tilt  = [face_tilt face_floor];

  norm_floor = [0 0;...
                1 1;...
                0 0];
  norm_tilt  = [norm_tilt norm_floor];

  [DpthImg,IR_now,IR_bin,IR_ref,IR_ind] = vc_KinectSimulator_Depth(vertex_tilt,face_tilt,norm_tilt,...
      'default','default','default',[]);
      
  save -mat7-binary 'IrBin.mat' 'IR_bin'
  save -mat7-binary 'IrNow.mat' 'IR_now'
  save -mat7-binary 'RefImgs.mat' 'IR_ref' 'IR_ind'
      
  figure, imshow(DpthImg,[])
  title('Noisy depth image of 60 degree tilted wall, center depth at 1000 mm')
  drawnow
end

%% DEMO BOX GRID ==========================================================
if true
  load('example_CAD/BoxGrid.mat')

  box_width = 200;  % mm
  box_depth = 500;  % mm
  wallDist  = 2000; % mm

  % Scale size of box and shift grid to further depth
  vertex_grid = vertex;
  vertex_grid(1:2,:) = vertex_grid(1:2,:).*box_width;
  vertex_grid(3,:) = vertex_grid(3,:).*box_depth + wallDist-box_depth;
  face_grid = face;
  norm_grid = normalf;
  GridRng = [min(vertex_grid(3,:)) max(vertex_grid(3,:))];

  [DpthImg,IR_now,IR_bin,IR_ref,IR_ind] = vc_KinectSimulator_Depth(vertex_grid,face_grid,norm_grid,...
    'simple','default','default',wallDist,'imgrng',GridRng,'quant11','off',...
    'quant10','off','displayIR','off');
      
  save 'IrBin.mat' 'IR_bin'
  save 'IrNow.mat' 'IR_now'
  %save -mat7-binary 'RefImgs.mat' 'IR_ref' 'IR_ind'
  
  figure, imshow(DpthImg,[])
  title('Noisy depth image of box grid in front of flat wall')
  drawnow
end

%% DEMO TEAPOT ============================================================
if false
  load('example_CAD\Teapot.mat')

  % Shift CAD model to further depth
  vertex_teapot = vertex;
  
  vertex_teapot(1,:) = vertex_teapot(1,:) * 5;
  vertex_teapot(2,:) = vertex_teapot(2,:) * 5;
  vertex_teapot(3,:) = vertex_teapot(3,:) * 5 + 1600;
  
  face_teapot = face;
  norm_teapot = normalf;

  [DpthImg,IR_now,IR_bin,IR_ref,IR_ind] = vc_KinectSimulator_Depth(vertex_teapot,face_teapot,norm_teapot,...
    'default','default','default',[]);
      
  save -mat7-binary 'IrBin.mat' 'IR_bin'
  save -mat7-binary 'IrNow.mat' 'IR_now'
  save -mat7-binary 'RefImgs.mat' 'IR_ref' 'IR_ind'
  
  figure, imshow(DpthImg,[])
  title('Noisy depth image of teapot in front of flat wall')
  drawnow
end

%% DEMO XBOX CONTROLLER ===================================================
if false
  load('example_CAD\XboxController.mat')

  % Shift CAD model to further depth
  vertex_xbox = vertex;
  
  vertex_xbox(1,:) = vertex_xbox(1,:) * 5;
  vertex_xbox(2,:) = vertex_xbox(2,:) * 5;
  vertex_xbox(3,:) = vertex_xbox(3,:) * 5 + 1500;
  
  face_xbox = face;
  norm_xbox = normalf;

  [DpthImg,IR_now,IR_bin,IR_ref,IR_ind] = vc_KinectSimulator_Depth(vertex_xbox,face_xbox,norm_xbox,...
      'default','default','default',[]);
      
  save -mat7-binary 'IrBin.mat' 'IR_bin'
  save -mat7-binary 'IrNow.mat' 'IR_now'
  save -mat7-binary 'RefImgs.mat' 'IR_ref' 'IR_ind'
      
  figure, imshow(DpthImg,[])
  title('Noisy depth image of xbox controller in front of nothing')
  drawnow
end