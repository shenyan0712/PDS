#pragma once

#define VEC_WIDTH			1									//vector width， 经测试，1对R9 290为最佳
#define TileSize			15									//经测试，11对R9 290为最佳
#define TileSize_C			(TileSize/VEC_WIDTH)				//tile size in columns
#define TileSize_R			TileSize									//tile size in rows

/*
#define WorkPerThrd_R		4									//work per thread in rows
#define WorkPerThrd_C		(WorkPerThrd_R/VEC_WIDTH)			//work per thread in columns
#define ThrdPerTile_R		(TileSize_R/WorkPerThrd_R)			//thread per tile in rows
#define ThrdPerTile_C		(TileSize_C/WorkPerThrd_C)			//step unit in columns
#define WorkPerTile_R		(WorkPerThrd_R*ThrdPerTile_R)
#define WorkPerTile_C		(WorkPerThrd_C*ThrdPerTile_C)
*/

#if VEC_WIDTH==1
#define VEC double
#elif VEC_WIDTH==2
#define VEC	double2
#elif VEC_WIDTH==4
#define VEC double4
#endif