%mex -c -v -I../opcode -I../opcode/Ice
mex -I../opcode -I../opcode/Ice opcodemeshmex.cpp ...
           ../opcode/OPC_AABBCollider.cpp          ...
		   ../opcode/OPC_AABBTree.cpp              ...
		   ../opcode/OPC_BaseModel.cpp             ...
		   ../opcode/OPC_BoxPruning.cpp            ...
		   ../opcode/OPC_Collider.cpp              ...
		   ../opcode/OPC_Common.cpp                ...
		   ../opcode/OPC_HybridModel.cpp           ...
		   ../opcode/OPC_LSSCollider.cpp           ...
		   ../opcode/OPC_MeshInterface.cpp         ...
		   ../opcode/OPC_Model.cpp	               ...
		   ../opcode/OPC_OBBCollider.cpp           ...
		   ../opcode/Opcode.cpp                    ...
		   ../opcode/OPC_OptimizedTree.cpp         ...
		   ../opcode/OPC_Picking.cpp               ...
		   ../opcode/OPC_PlanesCollider.cpp        ...
		   ../opcode/OPC_RayCollider.cpp           ...
		   ../opcode/OPC_SphereCollider.cpp	       ...
		   ../opcode/OPC_SweepAndPrune.cpp         ...
		   ../opcode/OPC_TreeBuilders.cpp          ...
		   ../opcode/OPC_TreeCollider.cpp          ...
		   ../opcode/OPC_VolumeCollider.cpp        ...
		   ../opcode/StdAfx.cpp                    ...
		   ../opcode/Ice/IceAABB.cpp               ...
		   ../opcode/Ice/IceContainer.cpp          ...
		   ../opcode/Ice/IceHPoint.cpp             ...
		   ../opcode/Ice/IceIndexedTriangle.cpp    ...
		   ../opcode/Ice/IceMatrix3x3.cpp          ...
		   ../opcode/Ice/IceMatrix4x4.cpp          ...
		   ../opcode/Ice/IceOBB.cpp                ...
		   ../opcode/Ice/IcePlane.cpp              ...
		   ../opcode/Ice/IcePoint.cpp              ...
		   ../opcode/Ice/IceRandom.cpp             ...
		   ../opcode/Ice/IceRay.cpp                ...
		   ../opcode/Ice/IceRevisitedRadix.cpp     ...
		   ../opcode/Ice/IceSegment.cpp            ...
		   ../opcode/Ice/IceTriangle.cpp           ...
		   ../opcode/Ice/IceUtils.cpp              



movefile(sprintf('opcodemeshmex.%s',mexext),'../matlab');