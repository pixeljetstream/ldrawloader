/**
 * Copyright (c) 2019-2020 Christoph Kubisch
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

// Report bugs and download new versions at https://github.com/pixeljetstream/ldrawloader

#pragma once

#include <stdint.h>

// LDR_CFG_THREADSAFE
//  Enabled: you can load parts/models from multiple threads
//  as well as build renderparts. The library will wait via yields
//  on depending operations to have completed.

#ifndef LDR_CFG_THREADSAFE
#define LDR_CFG_THREADSAFE 1
#endif

// LDR_CFG_C_API
//  if disabled must include ldrawloader.hpp
#ifndef LDR_CFG_C_API
#define LDR_CFG_C_API 1
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************
  These files implement a basic ldraw (.ldr/.mpd) file loader.
  You need to have a ldraw part library (http://www.ldraw.org/)
  installed and pass the path at creation time of the loader.

  To convert from LDD to ldraw, make use of
  https://www.eurobricks.com/forum/index.php?/forums/topic/137193-more-up-to-date-ldrawxml-lddldraw-conversion-file/
  or
  https://gitlab.com/sylvainls/lxf2ldr.html

  C interface exists, but you can also use the hpp implementation
  directly.

  Functionality:
    Model/Part -> basic ldraw part/model definitions
                  parts can be "fixed" (t-junctions, non-manifold etc.)

    RenderModel/RenderPart -> meant for rendering, per-vertex normal splitting
                              optional chamfered hard edges.
                              operates on "fixed parts" (implicit or explicit)
  TODO
  - improve hard edge merging/edge collapse for watertightness
    currently some models have adjacent "rings" with different radial subdivision,
    meaning every N th vertex is the same, but in-between there are gaps
  - lsynth support http://lsynth.sourceforge.net/LSynth.html
  - use ngons in mesh/base parts
  - chamfers can cause edges to be hidden, need to add chamfered lines
  - joining of intersecting/adjacent surfaces to create true solid 
    (very often open-edges are on top of other surfaces)
  - render linestrip extraction (detection of line curves)
  - binary cachefile
****************************************************/

// if you want to wrap this into dll etc.
#ifndef LDR_API
#define LDR_API
#endif

#define LDR_LOADER_VERSION_MAJOR 0
#define LDR_LOADER_VERSION_MINOR 3
#define LDR_LOADER_VERSION_CACHE 0

#define LDR_INVALID_ID (~0)
#define LDR_INVALID_IDX (~0)
#define LDR_INVALID_SHAPETYPE 0

#define LDR_RESTRICT __restrict

typedef enum LdrResult : int32_t
{
  LDR_WARNING_IN_USE         = 2,
  LDR_WARNING_PART_NOT_FOUND = 1,

  LDR_SUCCESS = 0,

  LDR_ERROR_FILE_NOT_FOUND      = -1,
  LDR_ERROR_PARSER              = -2,
  LDR_ERROR_INVALID_OPERATION   = -3,
  LDR_ERROR_RESERVED_MEMORY     = -4,
  LDR_ERROR_INITIALIZATION      = -5,
  LDR_ERROR_DEPENDENT_OPERATION = -6,
  LDR_ERROR_OTHER               = -7,
} LdrResult;

typedef enum LdrBool32 : uint32_t
{
  LDR_FALSE = 0,
  LDR_TRUE  = 1,
} LdrBool32;

typedef enum LdrObjectType : uint32_t
{
  LDR_OBJECT_PRIMITIVE,
  LDR_OBJECT_PART,
  LDR_OBJECT_MODEL,
} LdrObjectType;

typedef uint32_t LdrPartID;
typedef uint32_t LdrPrimitiveID;
typedef uint32_t LdrMaterialID;
typedef uint64_t LdrShapeType;

typedef uint32_t LdrVertexIndex;

// LdrRawData is used within structs that contain pointers themselves.
// All these pointers use a continuous allocation within raw::data.
// This makes serialization easier.

typedef struct LdrRawData
{
  size_t size;
  void*  data;
} LdrRawData;

typedef struct LdrMatrix
{
  union
  {
    float values[16];
    float col[4][4];
  };
} LdrMatrix;

typedef struct LdrVector
{
  union
  {
    struct
    {
      float x;
      float y;
      float z;
    };
    float vec[3];
  };
} LdrVector;

typedef struct LdrBbox
{
  LdrVector min;
  LdrVector max;
} LdrBbox;

//////////////////////////////////////////////////////////////////////////

// materials with code >= 0x2000000 are kept as is
// materials with code >= 10000 are rebased and start at 512

typedef enum LdrMaterialSpecial : uint32_t
{
  LDR_MATERIALID_INHERIT     = 16,
  LDR_MATERIALID_EDGE        = 24,
  // materialIds starting from this value are to be interpreted as 0x00RRGGBB
  // about 100 part files use this deprecated method
  LDR_MATERIALID_DIRECTSTART = 0x2000000,
} LdrMaterialSpecial;

typedef enum LdrMaterialType : uint32_t
{
  LDR_MATERIAL_SOLID,
  LDR_MATERIAL_TRANSPARENT,
  LDR_MATERIAL_RUBBER,
  LDR_MATERIAL_CHROME,
  LDR_MATERIAL_METALLIC,
  LDR_MATERIAL_SPECKLE,
  LDR_MATERIAL_PEARL,
  LDR_MATERIAL_GLITTER,
  LDR_MATERIAL_MILKY,
  LDR_MATERIAL_COMMON,
  LDR_MATERIAL_DEFAULT,
} LdrMaterialType;

typedef struct LdrMaterial
{
  uint8_t         baseColor[3];
  uint8_t         alpha;
  uint8_t         edgeColor[3];
  uint8_t         emissive;
  LdrMaterialType type;
  char            name[48];
  union
  {
    struct
    {
      uint8_t color[3];
      float   fraction;
      float   vfraction;
      float   size;
    } glitter;
    struct
    {
      uint8_t color[3];
      float   fraction;
      float   minsize;
      float   maxsize;
    } speckle;
  };
} LdrMaterial;

//////////////////////////////////////////////////////////////////////////

typedef struct LdrPartFlag
{
  // primitives are loaded from ldraw/p directory, basic shapes for merging
  uint32_t isPrimitive : 1;
  // subparts are loaded from the ldraw/parts/s directory, merged in similar to primitives, but more complex
  uint32_t isSubpart : 1;
  // part doesn't inherit all materials, but has some per-triangle materials
  uint32_t hasComplexMaterial : 1;
  // part cannot use backface culling
  uint32_t hasNoBackFaceCulling : 1;
  // part can be chamfered (otherwise topology is too problematic)
  uint32_t canChamfer : 1;
  // had errors during fix operations
  uint32_t hasFixErrors : 1;
} LdrPartFlag;


// LdrShapes are expressed procedurally by the user of the library.
// Register shape types to do part/primitive replacement.
// Instead of the ldraw files being loaded,
// shapes are appended to a part's shape list.
// You are responsible to handle drawing/processing the shapes
// after loading yourself.

typedef struct LdrShape
{
  LdrShapeType  type;
  uint32_t      bfcInvert;
  LdrMaterialID material;
  uint32_t      _pad1;
  LdrMatrix     transform;
} LdrShape;

// LdrInstance references another LdrPart.
// An instance may override the material and provides the matrix
// transform.

typedef struct LdrInstance
{
  LdrMatrix     transform;
  LdrPartID     part;
  LdrMaterialID material;
  uint32_t      _pad1[2];  // for 16 byte aligned transform
} LdrInstance;

// LdrParts represent the geometry of individual parts.
// An LdrPart may be either referenced via LdrPartID or LdrPrimitiveID.
// Primitives are used to assemble parts from some basic components
// (e.g. studs) that are commonly found within parts. Their geometry
// is always merged into a part's geometry, or replaced by LdrShapes.

typedef struct LdrPart
{
  LdrResult loadResult;

  LdrPartFlag flag;
  LdrBbox     bbox;
  float       minEdgeLength;

  uint32_t num_positions;
  uint32_t num_lines;
  uint32_t num_optional_lines;
  uint32_t num_triangles;
  uint32_t num_shapes;
  uint32_t num_instances;
  uint32_t num_name;

  // all pointers are within raw
  LdrVector* LDR_RESTRICT      positions;
  LdrVertexIndex* LDR_RESTRICT lines;
  LdrVertexIndex* LDR_RESTRICT optional_lines;
  LdrVertexIndex* LDR_RESTRICT triangles;
  LdrVertexIndex* LDR_RESTRICT connections;  // per-vertex, if != ~0 means index of opposing non-manifold vertex split
  LdrMaterialID* LDR_RESTRICT  materials;
  uint32_t* LDR_RESTRICT       quads;  // per-traingle, if != ~0 means index of starting triangle
  LdrShape* LDR_RESTRICT       shapes;
  LdrInstance* LDR_RESTRICT    instances;  // sub-parts
  char* LDR_RESTRICT           name;

  LdrRawData raw;
} LdrPart;

// LdrModel contains a flattened lsit of LdrInstances of
// LdrParts as well as the LdrParts referenced by those parts.

typedef struct LdrModel
{
  uint32_t num_instances;
  LdrBbox  bbox;

  // all pointers are within raw
  // all instances are flattened containing sub-part instances
  LdrInstance* LDR_RESTRICT instances;

  LdrRawData raw;
} LdrModel;

//////////////////////////////////////////////////////////////////////////

typedef struct LdrRenderVertex
{
  LdrVector position;
  //LdrMaterialID material; // validity depends on LdrLoaderCreateInfo::renderpartVertexMaterials
  LdrVector normal;
  //uint32_t      _pad;
} LdrRenderVertex;

// LdrRenderPart is the renderable version of a (non-Primitive) LdrPart.
// After building a render part, it is self-sufficient and doesn't depend on the LdrPart.
// It uses the LdrRenderVertex, which provides smooth vertex normals.
// Depending on the configuration of the loader, it can contain
// a second set of triangles that represent the model with chamfered
// hard edges, for higher fidelity.

typedef struct LdrRenderPart
{
  // for render parts the materials
  // array only exists if flag.hasComplexMaterial

  // C suffix stands for chamfered

  LdrPartFlag flag;
  LdrBbox     bbox;

  uint32_t num_vertices;
  uint32_t num_lines;
  uint32_t num_triangles;
  uint32_t num_trianglesC;
  uint32_t num_shapes;

  // all pointers are within raw
  LdrRenderVertex* LDR_RESTRICT vertices;
  LdrVertexIndex* LDR_RESTRICT  lines;
  //LdrVertexIndex* LDR_RESTRICT  linesC; // TODO
  LdrVertexIndex* LDR_RESTRICT triangles;
  LdrVertexIndex* LDR_RESTRICT trianglesC;
  LdrMaterialID* LDR_RESTRICT  materials;
  LdrMaterialID* LDR_RESTRICT  materialsC;
  LdrShape* LDR_RESTRICT       shapes;

  LdrRawData raw;
} LdrRenderPart;

// LdrRenderInstance is used by the LdrModel
// to represent all LdrRenderParts of that model.

typedef struct LdrRenderInstance
{
  LdrInstance instance;
} LdrRenderInstance;

// LdrRenderModel contains all information
// to render a LdrModel. It does not depend
// on the original LdrModel, nor LdrParts.

typedef struct LdrRenderModel
{
  uint32_t num_instances;
  LdrBbox  bbox;

  // all pointers are within raw
  LdrRenderInstance* LDR_RESTRICT instances;

  LdrRawData raw;
} LdrRenderModel;


//////////////////////////////////////////////////////////////////////////

typedef struct LdrLoader*     LdrLoaderHDL;
typedef const LdrModel*       LdrModelHDL;
typedef const LdrRenderModel* LdrRenderModelHDL;

typedef enum LdrPartFixMode : uint32_t
{
  // Part fixing involves resolving t-junction, non-manifold surfaces etc.
  // It is required for building LdrRenderParts but can be done at their build time indirectly
  // or in advance on the LdrPart itself, in that case the original LdrPart topology is lost,
  // hence not recommended.

  LDR_PART_FIX_NONE,    // part fixing is never applied to LdrParts (temp fix is triggered for renderbuild)
  LDR_PART_FIX_ONLOAD,  // part fixing is done at load time to LdrParts
} LdrPartFixMode;

typedef enum LdrRenderPartBuildMode : uint32_t
{
  LDR_RENDERPART_BUILD_ONDEMAND,  // either via ldrBuildRenderParts or ldrCreateRenderModel with autoResolve
  LDR_RENDERPART_BUILD_ONLOAD,    // done immediately at part load time
} LdrRenderPartBuildMode;

typedef struct LdrLoaderCreateInfo
{
  // path of the ldraw library installation
  const char* basePath;
  // path for cachefiles to speed up loading (NYI)
  // config below must match as well as version are tested prior use
  const char* cacheFile;

  LdrPartFixMode partFixMode;
  LdrBool32      partFixTjunctions;    // required for chamfer
  LdrBool32      partHiResPrimitives;  // substitutes with /p/48 if possible

  // TODO allow pervertex splits based on materials
  //LdrBool32 renderpartTriangleMaterials;  // keeps triangle materials array
  //LdrBool32 renderpartVertexMaterials;    // split vertices on material edges, not yet implemented

  LdrRenderPartBuildMode renderpartBuildMode;

  // chamfers hard edges if possible, only valid if partFixTjunctions was true
  // 1 LDU ~ 0.4mm -> 0.1f works okay for chamfer
  float renderpartChamfer;
} LdrLoaderCreateInfo;

#if LDR_CFG_C_API

LDR_API void ldrGetDefaultCreateInfo(LdrLoaderCreateInfo* info);

LDR_API LdrResult ldrCreateLoader(const LdrLoaderCreateInfo* info, LdrLoaderHDL* pLoader);
LDR_API void      ldrDestroyLoader(LdrLoaderHDL loader);  // loader can be null

// override part with custom procedural type
LDR_API LdrResult ldrRegisterShapeType(LdrLoaderHDL loader, const char* filename, LdrShapeType type);

// override part with custom definitions
// all overrides should be done prior any model creation operations
// part.raw must be allocated by library and registration passes memory ownership to library
LDR_API LdrResult ldrRegisterPart(LdrLoaderHDL loader, const char* filename, const LdrPart* part, LdrPartID* pPartID);
LDR_API LdrResult ldrRegisterPrimitive(LdrLoaderHDL loader, const char* filename, const LdrPart* part);
LDR_API LdrResult ldrRegisterRenderPart(LdrLoaderHDL loader, LdrPartID partId, const LdrRenderPart* part);
// for custom overrides
LDR_API LdrResult ldrRawAllocate(LdrLoaderHDL loader, size_t size, LdrRawData* raw);
LDR_API LdrResult ldrRawFree(LdrLoaderHDL loader, const LdrRawData* raw);

// pre-load a part
LDR_API LdrResult ldrPreloadPart(LdrLoaderHDL loader, const char* filename, LdrPartID* pPartID);

// When "autoResolve" is used, all dependencies (part/primitive loading) are resolved automatically.
// Without this we defer loading the actual parts and you must load them manually.
LDR_API LdrResult ldrCreateModel(LdrLoaderHDL loader, const char* filename, LdrBool32 autoResolve, LdrModelHDL* pModel);
LDR_API void      ldrDestroyModel(LdrLoaderHDL loader, LdrModelHDL model);  // model can be null
// only required if autoResolve was false, all dependent deferred parts must have been loaded
LDR_API void ldrResolveModel(LdrLoaderHDL loader, LdrModelHDL model);

// When "autoResolve" is used, all dependencies (renderpart building) are resolved automatically.
LDR_API LdrResult ldrCreateRenderModel(LdrLoaderHDL loader, LdrModelHDL model, LdrBool32 autoResolve, LdrRenderModelHDL* pRenderModel);
LDR_API void ldrDestroyRenderModel(LdrLoaderHDL loader, LdrRenderModelHDL renderModel);  // renderModel can be null

// Use parts == nullptr to operate on all currently loaded parts (overrides numParts).
// only legal for LDR_RENDERPART_BUILD_ONDEMAND
LDR_API LdrResult ldrBuildRenderParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride);

// Use parts == nullptr to operate on all currently loaded parts (overrides numParts).
LDR_API LdrResult ldrLoadDeferredParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride);

LDR_API LdrPartID      ldrFindPart(LdrLoaderHDL loader, const char* filename);
LDR_API LdrPrimitiveID ldrFindPrimitive(LdrLoaderHDL loader, const char* filename);

// can be used to distribute part loading across threads if autoResolve is false on model create
LDR_API uint32_t ldrGetNumRegisteredParts(LdrLoaderHDL loader);

LDR_API uint32_t ldrGetNumRegisteredMaterials(LdrLoaderHDL loader);

// will return nullptr if id is invalid or renderpart not built
LDR_API const LdrMaterial*   ldrGetMaterial(LdrLoaderHDL loader, LdrMaterialID id);
LDR_API const LdrPart*       ldrGetPart(LdrLoaderHDL loader, LdrPartID id);
LDR_API const LdrPart*       ldrGetPrimitive(LdrLoaderHDL loader, LdrPrimitiveID id);
LDR_API const LdrRenderPart* ldrGetRenderPart(LdrLoaderHDL loader, LdrPartID id);

#endif

#ifdef __cplusplus
}
#endif
