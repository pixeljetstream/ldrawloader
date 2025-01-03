/**
 * Copyright (c) 2019-2024 Christoph Kubisch
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

#include "ldrawloader.hpp"

#include <algorithm>
#include <assert.h>
#include <intrin.h>
#include <stdlib.h>

#define LDR_DEBUG_FLAG_FILELOAD 1
#define LDR_DEBUG_FLAG 0
#define LDR_DEBUG_PRINT_NON_MANIFOLDS 0

namespace ldr {

// ignores projective

inline LdrMatrix mat_identity()
{
  LdrMatrix out = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  return out;
}

inline LdrMatrix mat_mul(const LdrMatrix& a, const LdrMatrix& b)
{
  LdrMatrix out;
  out.values[0] = a.values[0] * b.values[0] + a.values[4] * b.values[1] + a.values[8] * b.values[2];
  out.values[1] = a.values[1] * b.values[0] + a.values[5] * b.values[1] + a.values[9] * b.values[2];
  out.values[2] = a.values[2] * b.values[0] + a.values[6] * b.values[1] + a.values[10] * b.values[2];
  out.values[3] = 0;

  out.values[4] = a.values[0] * b.values[4] + a.values[4] * b.values[5] + a.values[8] * b.values[6];
  out.values[5] = a.values[1] * b.values[4] + a.values[5] * b.values[5] + a.values[9] * b.values[6];
  out.values[6] = a.values[2] * b.values[4] + a.values[6] * b.values[5] + a.values[10] * b.values[6];
  out.values[7] = 0;

  out.values[8]  = a.values[0] * b.values[8] + a.values[4] * b.values[9] + a.values[8] * b.values[10];
  out.values[9]  = a.values[1] * b.values[8] + a.values[5] * b.values[9] + a.values[9] * b.values[10];
  out.values[10] = a.values[2] * b.values[8] + a.values[6] * b.values[9] + a.values[10] * b.values[10];
  out.values[11] = 0;

  out.values[12] = a.values[0] * b.values[12] + a.values[4] * b.values[13] + a.values[8] * b.values[14] + a.values[12];
  out.values[13] = a.values[1] * b.values[12] + a.values[5] * b.values[13] + a.values[9] * b.values[14] + a.values[13];
  out.values[14] = a.values[2] * b.values[12] + a.values[6] * b.values[13] + a.values[10] * b.values[14] + a.values[14];
  out.values[15] = 1;
  return out;
}

inline LdrVector transform_point(const LdrMatrix& transform, const LdrVector& vec)
{
  LdrVector    out;
  const float* mat = transform.values;
  out.x            = vec.x * (mat)[0] + vec.y * (mat)[4] + vec.z * (mat)[8] + (mat)[12];
  out.y            = vec.x * (mat)[1] + vec.y * (mat)[5] + vec.z * (mat)[9] + (mat)[13];
  out.z            = vec.x * (mat)[2] + vec.y * (mat)[6] + vec.z * (mat)[10] + (mat)[14];
  return out;
}

inline LdrVector transform_vec(const LdrMatrix& transform, const LdrVector& vec)
{
  LdrVector    out;
  const float* mat = transform.values;
  out.x            = vec.x * (mat)[0] + vec.y * (mat)[4] + vec.z * (mat)[8];
  out.y            = vec.x * (mat)[1] + vec.y * (mat)[5] + vec.z * (mat)[9];
  out.z            = vec.x * (mat)[2] + vec.y * (mat)[6] + vec.z * (mat)[10];
  return out;
}

inline float mat_determinant(const LdrMatrix& transform)
{
  // Sarrus rule
  return transform.col[0][0] * transform.col[1][1] * transform.col[2][2]
         + transform.col[1][0] * transform.col[2][1] * transform.col[0][2]
         + transform.col[2][0] * transform.col[0][1] * transform.col[1][2]
         - transform.col[2][0] * transform.col[1][1] * transform.col[0][2]
         - transform.col[0][0] * transform.col[2][1] * transform.col[1][2]
         - transform.col[1][0] * transform.col[0][1] * transform.col[2][2];
}

inline LdrVector make_vec(const float* v)
{
  return {v[0], v[1], v[2]};
}
inline LdrVector vec_min(const LdrVector a, const LdrVector b)
{
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline LdrVector vec_max(const LdrVector a, const LdrVector b)
{
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
inline LdrVector vec_add(const LdrVector a, const LdrVector b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline LdrVector vec_sub(const LdrVector a, const LdrVector b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline LdrVector vec_div(const LdrVector a, const LdrVector b)
{
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
inline LdrVector vec_mul(const LdrVector a, const LdrVector b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
inline LdrVector vec_mul(const LdrVector a, const float b)
{
  return {a.x * b, a.y * b, a.z * b};
}
inline LdrVector vec_ceil(const LdrVector a)
{
  return {ceilf(a.x), ceilf(a.y), ceilf(a.z)};
}
inline LdrVector vec_floor(const LdrVector a)
{
  return {floorf(a.x), floorf(a.y), floorf(a.z)};
}
inline LdrVector vec_clamp(const LdrVector a, const float lowerV, const float upperV)
{
  return {std::max(std::min(upperV, a.x), lowerV), std::max(std::min(upperV, a.y), lowerV), std::max(std::min(upperV, a.z), lowerV)};
}
inline LdrVector vec_cross(const LdrVector a, const LdrVector b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float vec_dot(const LdrVector a, const LdrVector b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float vec_sq_length(const LdrVector a)
{
  return vec_dot(a, a);
}
inline float vec_length(const LdrVector a)
{
  return sqrt(vec_dot(a, a));
}
inline LdrVector vec_normalize(const LdrVector a)
{
  float len = vec_length(a);
  return vec_mul(a, (1.0f / len));
}
inline LdrVector vec_normalize_length(const LdrVector a, float& length)
{
  float len = vec_length(a);
  length    = len;
  return vec_mul(a, (1.0f / len));
}
inline LdrVector vec_neg(const LdrVector a)
{
  return {-a.x, -a.y, -a.z};
}
inline void bbox_merge(LdrBbox& bbox, const LdrVector vec)
{
  bbox.min = vec_min(bbox.min, vec);
  bbox.max = vec_max(bbox.max, vec);
}

inline void bbox_merge(LdrBbox& bbox, const LdrMatrix& transform, const LdrBbox other)
{
  const float* values = &other.min.x;
  uint32_t     min    = 0;
  uint32_t     max    = uint32_t(offsetof(LdrBbox, max) / sizeof(float));
  for(int i = 0; i < 8; i++) {
    LdrVector corner;
    corner.x        = values[(i & 1 ? max : min) + 0];
    corner.y        = values[(i & 2 ? max : min) + 1];
    corner.z        = values[(i & 4 ? max : min) + 2];
    LdrVector point = transform_point(transform, corner);
    bbox_merge(bbox, point);
  }
}

//////////////////////////////////////////////////////////////////////////

const float Loader::COPLANAR_TRIANGLE_DOT = 0.998f;  // around 3 degrees
const float Loader::NO_AREA_TRIANGLE_DOT  = 0.9999f;
const float Loader::FORCED_HARD_EDGE_DOT  = 0.2f;
const float Loader::CHAMFER_PARALLEL_DOT  = 0.999f;
const float Loader::ANGLE_45_DOT          = 0.7071f;
const float Loader::MIN_MERGE_EPSILON     = 0.015f;  // 1 LDU ~ 0.4mm

static_assert(LDR_INVALID_ID == LDR_INVALID_IDX);

class Mesh
{
public:
  static const uint32_t VTX_BITS = 31;
  static const uint32_t INVALID  = uint32_t(~0);

  static_assert(INVALID == LDR_INVALID_IDX);

  typedef uint64_t edgeHash_t;

  struct VertexPair
  {
    uint32_t a;
    uint32_t b;
  };

  static edgeHash_t make_edgeHash(uint32_t vtxA, uint32_t vtxB)
  {
    union
    {
      uint64_t   u64;
      VertexPair pair;
    };

    if(vtxA < vtxB) {
      pair.a = vtxA;
      pair.b = vtxB;
    }
    else {
      pair.a = vtxB;
      pair.b = vtxA;
    }

    return u64;
  }

  struct Connectivity
  {
    static const uint32_t GROWTH = 16;

    struct Info
    {
      uint16_t count     = 0;
      uint16_t allocated = 0;
      uint32_t offset    = INVALID;
    };

    Loader::TVector<Info>     infos;
    Loader::TVector<uint32_t> content;

    Loader* loader = nullptr;  // mostly for debugging


    void resize(uint32_t num)
    {
      infos.resize(num);
      content.reserve(num * GROWTH);
    }

    // pointer only valid as long as no edits are made
    const uint32_t* getConnected(uint32_t v, uint32_t& count) const
    {
      count = infos[v].count;
      return content.data() + infos[v].offset;
    }

    void add(uint32_t v, uint32_t element)
    {
      uint32_t old    = infos[v].count++;
      uint32_t offset = infos[v].offset;
      if(infos[v].allocated < infos[v].count) {
        uint32_t newoffset = (uint32_t)content.size();
        infos[v].allocated += GROWTH;
        infos[v].offset = newoffset;
        content.reserve(std::max(content.size(), (content.size() * 3) / 2));
        content.resize(content.size() + infos[v].allocated);
        if(old) {
          memcpy(content.data() + newoffset, content.data() + offset, sizeof(uint32_t) * old);
        }

        offset = newoffset;
      }

      content[offset + old] = element;
    }

    void remove(uint32_t v, uint32_t element)
    {
      uint32_t* data  = content.data() + infos[v].offset;
      uint32_t  count = infos[v].count;
      for(uint32_t i = 0; i < count; i++) {
        if(data[i] == element) {
          data[i] = data[count - 1];
          infos[v].count--;
        }
      }
    }
  };

  struct TriAdjacency
  {
    // other triangle per edge
    uint32_t edgeSmoothTri[3]   = {INVALID, INVALID, INVALID};
    uint32_t edgeManifoldTri[3] = {INVALID, INVALID, INVALID};
  };

  // additional left/right for non-manifold overflow
  // linked list
  struct EdgeNM
  {
    uint32_t tri    = INVALID;
    uint32_t nextNM = INVALID;
    bool     isLeft = true;
    float    angle  = 0;
  };

  struct Edge
  {
    // left triangle has its edge running from A to B
    // right triangle in opposite direction.

    uint32_t vtxA;
    uint32_t vtxB;
    uint32_t triLeft;
    uint32_t triRight;
    float    angleRight;
    uint32_t flag;
    uint32_t nmList;

    bool isNonManifold() const { return (nmList != INVALID); }

    bool isDead() const { return triLeft == INVALID; }

    bool isOpen() const { return triRight == INVALID; }

    bool hasFace(uint32_t idx) const { return triLeft == idx || triRight == idx; }

    uint32_t otherVertex(uint32_t idx) const { return idx != vtxA ? vtxA : vtxB; }

    uint32_t otherTri(uint32_t idx) const { return idx != triLeft ? triLeft : triRight; }

    uint32_t getTri(uint32_t right) const { return right ? triRight : triLeft; }

    uint32_t getVertex(uint32_t b) const { return b ? vtxB : vtxA; }

    VertexPair getVertexPair() const { return {vtxA, vtxB}; }
  };


  uint32_t numVertices              = 0;
  uint32_t numTriangles             = 0;
  uint32_t numEdges                 = 0;
  uint32_t freeNM                   = INVALID;
  bool     requiresVertexTriangles  = false;
  bool     nonManifoldEdges         = false;
  bool     nonManifoldNeedsOrdering = false;
  bool     hasOverlap               = false;

  uint32_t* triangles = nullptr;

  Connectivity vtxTriangles;
  Connectivity vtxEdges;

  Loader::TVector<TriAdjacency> triAdjacencies;
  Loader::BitArray              triAlive;

  Loader::TVector<Edge>   edges;
  Loader::TVector<EdgeNM> edgesNM;

  std::unordered_map<edgeHash_t, uint32_t> lookupEdge;

  Loader* loader = nullptr;  // for debugging

  inline bool resizeVertices(uint32_t num)
  {
    numVertices = num;
    vtxEdges.resize(numVertices);

    if(requiresVertexTriangles) {
      vtxTriangles.resize(numVertices);
    }

    if(num > (1 << VTX_BITS)) {
      // FIXME check against this error
      fprintf(stderr, "Mesh VTX_BITS too small - %d > %d\n", num, (1 << VTX_BITS));
      exit(-1);
      return true;
    }

    return false;
  }

  inline bool getVertexClosable(uint32_t vertex)
  {
    uint32_t        open = 0;
    uint32_t        count;
    const uint32_t* curEdges = vtxEdges.getConnected(vertex, count);
    for(uint32_t e = 0; e < count; e++) {
      open += edges[curEdges[e]].isOpen() && !edges[curEdges[e]].isDead();
    }
    return open == 2;
  }

  inline Edge* getEdge(uint32_t vtxA, uint32_t vtxB)
  {
    const auto it = lookupEdge.find(make_edgeHash(vtxA, vtxB));
    if(it != lookupEdge.cend()) {
      return &edges[it->second];
    }
    return nullptr;
  }

  inline const Edge* getEdge(uint32_t vtxA, uint32_t vtxB) const
  {
    const auto it = lookupEdge.find(make_edgeHash(vtxA, vtxB));
    if(it != lookupEdge.cend()) {
      return &edges[it->second];
    }
    return nullptr;
  }

  inline uint32_t getEdgeIdx(uint32_t vtxA, uint32_t vtxB) const
  {
    const auto it = lookupEdge.find(make_edgeHash(vtxA, vtxB));
    if(it != lookupEdge.cend()) {
      return it->second;
    }
    return INVALID;
  }

  inline uint32_t* getTriangle(uint32_t t) { return &triangles[t * 3]; }

  inline const uint32_t* getTriangle(uint32_t t) const { return &triangles[t * 3]; }

  inline VertexPair getTriangleEdgeVertices(uint32_t t, uint32_t e) const
  {
    return {triangles[t * 3 + e], triangles[t * 3 + (e + 1) % 3]};
  }

  inline void setTriSmoothAdjacency(uint32_t t, uint32_t e, uint32_t tOther)
  {
    assert(t != tOther);
    assert(triAdjacencies[t].edgeSmoothTri[e] == INVALID || triAdjacencies[t].edgeSmoothTri[e] == tOther);
    triAdjacencies[t].edgeSmoothTri[e] = tOther;
  }
  inline void setTriManifoldAdjacency(uint32_t t, uint32_t e, uint32_t tOther)
  {
    assert(t != tOther);
    assert(triAdjacencies[t].edgeManifoldTri[e] == INVALID || triAdjacencies[t].edgeManifoldTri[e] == tOther);
    triAdjacencies[t].edgeManifoldTri[e] = tOther;
  }
  bool areTriManifoldAdjacent(uint32_t ta, uint32_t tb) const
  {
    const TriAdjacency triA = triAdjacencies[ta];

    return (triA.edgeManifoldTri[0] == tb || triA.edgeManifoldTri[1] == tb || triA.edgeManifoldTri[2] == tb);
  }

  bool areTriSmoothAdjacent(uint32_t ta, uint32_t tb) const
  {
    const TriAdjacency triA = triAdjacencies[ta];

    return (triA.edgeSmoothTri[0] == tb || triA.edgeSmoothTri[1] == tb || triA.edgeSmoothTri[2] == tb);
  }

  static inline uint32_t findLowest(const uint32_t* indices)
  {
    uint32_t lowestIdx = indices[0];
    uint32_t lowest    = 0;
    if(indices[1] < lowestIdx) {
      lowestIdx = indices[1];
      lowest    = 1;
    }
    if(indices[2] < lowestIdx) {
      lowestIdx = indices[2];
      lowest    = 2;
    }

    return lowest;
  }

  inline bool areTrianglesSame(uint32_t t1, uint32_t t2) const
  {
    const uint32_t* indices1 = &triangles[t1 * 3];
    const uint32_t* indices2 = &triangles[t2 * 3];

    uint32_t lowest1 = findLowest(indices1);
    uint32_t lowest2 = findLowest(indices2);

    for(uint32_t i = 0; i < 3; i++) {
      if(indices1[(lowest1 + i) % 3] != indices2[(lowest2 + i) % 3])
        return false;
    }

    return true;
  }

  // find cross product of vectors with widest angle
  inline LdrVector getTriangleNormal(uint32_t t, const LdrVector* positions) const
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    LdrVector sides[3];
    float     minAngle = FLT_MAX;
    uint32_t  corner   = 0;

    for(uint32_t i = 0; i < 3; i++) {
      sides[i] = vec_normalize(vec_sub(positions[triangles[t * 3 + i]], positions[triangles[t * 3 + (i + 1) % 3]]));
    }
    for(uint32_t i = 0; i < 3; i++) {
      float angle = fabsf(vec_dot(vec_neg(sides[i]), sides[(i + 1) % 3]));
      if(angle < minAngle) {
        minAngle = angle;
        corner   = i;
      }
    }

    return vec_normalize(vec_cross((sides[corner]), (sides[(corner + 1) % 3])));
    //return vec_normalize(vec_cross(sides[0], vec_neg(sides[2])));
  }

  inline void getTriangleBasis(uint32_t t, const LdrVector* positions, LdrVector basis[3]) const
  {
    basis[0] = vec_normalize(vec_sub(positions[triangles[t * 3 + 0]], positions[triangles[t * 3 + 1]]));
    basis[2] = getTriangleNormal(t, positions);
    basis[1] = vec_normalize(vec_cross(basis[0], basis[2]));
  }

  // average of the 3 normals
  inline LdrVector getTriangleCross(uint32_t t, const LdrVector* positions) const
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    LdrVector sides[3];
    LdrVector cross = {0.0f, 0.0f, 0.0f};

    for(uint32_t i = 0; i < 3; i++) {
      sides[i] = (vec_sub(positions[triangles[t * 3 + i]], positions[triangles[t * 3 + (i + 1) % 3]]));
    }
    for(uint32_t i = 0; i < 3; i++) {
      cross = vec_add(cross, vec_cross((sides[i]), (sides[(i + 1) % 3])));
    }

    return vec_mul(cross, 1.0f / 3.0f);
  }

  inline const uint32_t getTriangleOtherVertex(uint32_t t, const Edge& edge) const
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    if(idxA != edge.vtxA && idxA != edge.vtxB)
      return idxA;
    if(idxB != edge.vtxA && idxB != edge.vtxB)
      return idxB;
    if(idxC != edge.vtxA && idxC != edge.vtxB)
      return idxC;

    return INVALID;
  }

  inline uint32_t findTriangleVertex(uint32_t t, uint32_t vtx) const
  {
    const uint32_t* indices = getTriangle(t);
    if(indices[0] == vtx)
      return 0;
    if(indices[1] == vtx)
      return 1;
    if(indices[2] == vtx)
      return 2;
    return INVALID;
  }

  inline void replaceTriangleVertex(uint32_t t, uint32_t vtx, uint32_t newVtx)
  {
    uint32_t* indices = getTriangle(t);
    if(indices[0] == vtx)
      indices[0] = newVtx;
    if(indices[1] == vtx)
      indices[1] = newVtx;
    if(indices[2] == vtx)
      indices[2] = newVtx;
  }

  inline uint32_t findTriangleConnectingVertices(uint32_t tA, uint32_t tB, uint32_t* vertsA, uint32_t* vertsB) const
  {
    const uint32_t* indices = getTriangle(tA);

    uint32_t num = 0;
    uint32_t idxB;

    idxB = findTriangleVertex(tB, indices[0]);
    if(idxB != INVALID) {
      vertsA[num] = 0;
      vertsB[num] = idxB;
      num++;
    }

    idxB = findTriangleVertex(tB, indices[1]);
    if(idxB != INVALID) {
      vertsA[num] = 1;
      vertsB[num] = idxB;
      num++;
    }

    idxB = findTriangleVertex(tB, indices[2]);
    if(idxB != INVALID) {
      vertsA[num] = 2;
      vertsB[num] = idxB;
      num++;
    }

    return num;
  }

  uint32_t addEdgeNM(uint32_t startNM, uint32_t tri, bool isLeft, bool& identicalNeighbor)
  {
    uint32_t nextNM = startNM;

    while(nextNM != INVALID) {
      EdgeNM& edgeNM = edgesNM[nextNM];
      if(edgeNM.tri == tri)
        return startNM;
      if(areTrianglesSame(edgeNM.tri, tri)) {
        identicalNeighbor = true;
        return startNM;
      }

      nextNM = edgeNM.nextNM;
    }

    uint32_t outNextNM;
    if(freeNM != INVALID) {
      outNextNM = freeNM;
      freeNM    = edgesNM[freeNM].nextNM;

      edgesNM[outNextNM].tri    = tri;
      edgesNM[outNextNM].nextNM = startNM;
      edgesNM[outNextNM].isLeft = isLeft;
    }
    else {
      outNextNM = edgesNM.size();
      edgesNM.push_back({tri, startNM, isLeft, 0});
    }

    return outNextNM;
  }

  uint32_t findFirstEdgeNM(uint32_t nextNM, bool isLeft) const
  {
    while(nextNM != INVALID) {
      const EdgeNM& edgeNM = edgesNM[nextNM];
      if(edgeNM.isLeft == isLeft) {
        return edgeNM.tri;
      }
      nextNM = edgeNM.nextNM;
    }

    return INVALID;
  }

  const EdgeNM* iterateEdgeNM(uint32_t& startNM) const
  {
    if(startNM != INVALID) {
      uint32_t current = startNM;
      startNM          = edgesNM[startNM].nextNM;
      return &edgesNM[current];
    }
    return nullptr;
  }

  EdgeNM* iterateEdgeNM(uint32_t& startNM)
  {
    if(startNM != INVALID) {
      uint32_t current = startNM;
      startNM          = edgesNM[startNM].nextNM;
      return &edgesNM[current];
    }
    return nullptr;
  }

  bool removeEdgeNM(uint32_t& startNM, uint32_t tri)
  {
    uint32_t nextNM = startNM;
    uint32_t prevNM = INVALID;

    while(nextNM != INVALID) {
      EdgeNM& edgeNM = edgesNM[nextNM];
      if(edgeNM.tri == tri) {
        uint32_t oldNextNM = edgeNM.nextNM;
        // insert into freelist
        edgeNM.nextNM = freeNM;
        edgeNM.tri    = INVALID;
        freeNM        = nextNM;

        if(prevNM != INVALID) {
          edgesNM[prevNM].nextNM = oldNextNM;
        }
        else {
          // change list head
          startNM = oldNextNM;
        }

        return true;
      }

      prevNM = nextNM;
      nextNM = edgeNM.nextNM;
    }
    return false;
  }

  void addEdge(uint32_t vtxA, uint32_t vtxB, uint32_t tri, uint32_t& nonManifold, bool& identicalNeighbor)
  {
    if(identicalNeighbor)
      return;

    edgeHash_t edgeHash = make_edgeHash(vtxA, vtxB);
    auto       it       = lookupEdge.find(edgeHash);

    if(it != lookupEdge.end()) {
      Edge& edge = edges[it->second];

      if(edge.triLeft == tri || edge.triRight == tri) {
        it = it;
      }
      else if(vtxA == edge.vtxB && edge.triRight == INVALID) {
        if(areTrianglesSame(edge.triLeft, tri)) {
          identicalNeighbor = true;
          return;
        }
        edge.triRight = tri;
      }
      else {
        edge.nmList = addEdgeNM(edge.nmList, tri, edge.vtxA == vtxA, identicalNeighbor);
        if(identicalNeighbor)
          return;

        nonManifold              = it->second;
        nonManifoldNeedsOrdering = true;
      }

      return;
    }
    else {
      uint32_t edgeIdx = numEdges;
      Edge     edge;
      edge.vtxA       = vtxA;
      edge.vtxB       = vtxB;
      edge.triLeft    = tri;
      edge.triRight   = INVALID;
      edge.angleRight = 0;
      edge.flag       = 0;
      edge.nmList     = INVALID;

      edges.push_back(edge);
      lookupEdge.insert({edgeHash, edgeIdx});

      vtxEdges.add(vtxA, edgeIdx);
      vtxEdges.add(vtxB, edgeIdx);

      numEdges++;
      return;
    }
  }

  inline void removeEdge(uint32_t vtxA, uint32_t vtxB, uint32_t tri)
  {
    edgeHash_t edgeHash = make_edgeHash(vtxA, vtxB);
    auto       it       = lookupEdge.find(edgeHash);
    if(it == lookupEdge.end())
      return;

    Edge& edge = edges[it->second];
    if(!edge.isNonManifold()) {
      if(edge.triRight == tri) {
        edge.triRight = INVALID;
      }
      else if(edge.triRight != INVALID) {
        // if only right left, migrate right to left
        edge.triLeft  = edge.triRight;
        uint32_t temp = edge.vtxB;
        edge.vtxB     = edge.vtxA;
        edge.vtxA     = temp;
        edge.triRight = INVALID;
      }
      else {
        edge.triLeft = INVALID;

        vtxEdges.remove(edge.vtxA, it->second);
        vtxEdges.remove(edge.vtxB, it->second);

        lookupEdge.erase(edgeHash);
      }
    }
    else {
      bool isLeft   = edge.vtxA == vtxA;
      bool popFirst = edge.triLeft == tri || edge.triRight == tri;

      uint32_t first     = popFirst ? findFirstEdgeNM(edge.nmList, isLeft) : 0;
      uint32_t removeTri = tri;

      // find first with matching state
      if(edge.triLeft == tri) {
        edge.triLeft             = first;
        removeTri                = first;
        nonManifoldNeedsOrdering = true;
      }
      else if(edge.triRight == tri) {
        edge.triRight = first;
        removeTri     = first;
      }

      removeEdgeNM(edge.nmList, removeTri);

      // if only right exist, migrate right to left
      if(edge.triLeft == INVALID && edge.triRight != INVALID) {
        edge.triLeft  = edge.triRight;
        uint32_t temp = edge.vtxB;
        edge.vtxB     = edge.vtxA;
        edge.vtxA     = temp;
        edge.triRight = INVALID;
        // flip left/right in edgeNM list
        uint32_t nextNM = edge.nmList;
        while(nextNM != INVALID) {
          iterateEdgeNM(nextNM)->isLeft ^= true;
        }
      }
      else if(edge.triLeft == INVALID) {
        vtxEdges.remove(edge.vtxA, it->second);
        vtxEdges.remove(edge.vtxB, it->second);

        lookupEdge.erase(edgeHash);
      }
    }
  }

  inline uint32_t addTriangle(uint32_t t)
  {
    uint32_t nonManifold = INVALID;

    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    if(idxA == idxB || idxB == idxC || idxA == idxC)
      return INVALID;

    bool identicalNeighbor = false;

    addEdge(idxA, idxB, t, nonManifold, identicalNeighbor);
    addEdge(idxB, idxC, t, nonManifold, identicalNeighbor);
    addEdge(idxC, idxA, t, nonManifold, identicalNeighbor);

    if(requiresVertexTriangles && !identicalNeighbor) {
      vtxTriangles.add(idxA, t);
      vtxTriangles.add(idxB, t);
      vtxTriangles.add(idxC, t);
    }

    if(triAlive.size() <= t) {
      triAlive.resize(t + 1, false);
      numTriangles = t + 1;
    }

    assert(!triAlive.getBit(t));
    triAlive.setBit(t, !identicalNeighbor);

    return nonManifold;
  }

  inline void removeTriangle(uint32_t t, bool flagDead = true)
  {
    if(!triAlive.getBit(t))
      return;

    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    removeEdge(idxA, idxB, t);
    removeEdge(idxB, idxC, t);
    removeEdge(idxC, idxA, t);

    if(requiresVertexTriangles) {
      vtxTriangles.remove(idxA, t);
      vtxTriangles.remove(idxB, t);
      vtxTriangles.remove(idxC, t);
    }
    if(flagDead) {
      triAlive.setBit(t, false);
    }
  }

  void initBasics(uint32_t numV, uint32_t numT, uint32_t* tris, Loader* _loader)
  {
    loader              = _loader;
    vtxEdges.loader     = _loader;
    vtxTriangles.loader = _loader;


    numEdges     = 0;
    numTriangles = numT;
    triangles    = tris;

    edges.reserve(numT * 3);
    lookupEdge.reserve(numT * 3);
    triAlive.resize(numT, false);
    triAdjacencies.resize(numT, TriAdjacency());

    resizeVertices(numV);
  }

  bool initFull(uint32_t numV, uint32_t numT, uint32_t* tris, const LdrVector* positions, Loader* _loader)
  {
    requiresVertexTriangles = true;

    bool nonManifold = false;
    initBasics(numV, numT, tris, _loader);

    for(uint32_t t = 0; t < numT; t++) {
      nonManifold = (addTriangle(t) != INVALID) || nonManifold;
    }

    if(nonManifold) {
      orderNonManifoldByAngle(positions);
    }


    return nonManifold;
  }

  struct SortBasis
  {
    LdrVector posA;
    LdrVector vecTriNormal;    // perpendicular to triangle plane
    LdrVector vecTriOpenSide;  // in triangle plane, parallel to edge

    float getAngle(LdrVector posC) const
    {
      LdrVector vecCA = vec_normalize(vec_sub(posC, posA));
      float     y     = -vec_dot(vecTriNormal, vecCA);
      float     x     = vec_dot(vecTriOpenSide, vecCA);

      return atan2f(y, x);
    }

    void init(const LdrVector* positions, uint32_t a, uint32_t b, uint32_t c)
    {
      posA           = positions[a];
      LdrVector posB = positions[b];
      LdrVector posC = positions[c];

      LdrVector vecBA = vec_normalize(vec_sub(posB, posA));
      LdrVector vecCA = vec_normalize(vec_sub(posC, posA));
      vecTriNormal    = vec_normalize(vec_cross(vecBA, vecCA));
      vecTriOpenSide  = vec_normalize(vec_cross(vecBA, vecTriNormal));
    }
  };

  struct SortEdge
  {
    float    angle;
    uint32_t tri;
    bool     isLeft;

    static bool comparator(const SortEdge& a, const SortEdge& b) { return a.angle < b.angle; }
  };


  void orderNonManifoldByAngle(const LdrVector* positions)
  {
    if(!nonManifoldNeedsOrdering)
      return;

    Loader::TVector<SortEdge> sortEdges;
    sortEdges.reserve(128);

    for(uint32_t e = 0; e < numEdges; e++) {
      Edge& edge = edges[e];
      if(!(edge.isNonManifold()))
        continue;

      SortBasis basis;
      basis.init(positions, edge.vtxA, edge.vtxB, getTriangleOtherVertex(edge.triLeft, edge));

      sortEdges.clear();

      // compute angles and push into sorting queue

      if(edge.triRight != INVALID) {
        SortEdge sedge;
        sedge.angle  = basis.getAngle(positions[getTriangleOtherVertex(edge.triRight, edge)]);
        sedge.isLeft = false;
        sedge.tri    = edge.triRight;
        sortEdges.push_back(sedge);
      }

      uint32_t nextNM = edge.nmList;
      while(nextNM != Mesh::INVALID) {
        const EdgeNM& edgeNM = edgesNM[nextNM];
        SortEdge      sedge;
        sedge.angle  = basis.getAngle(positions[getTriangleOtherVertex(edgeNM.tri, edge)]);
        sedge.isLeft = edgeNM.isLeft;
        sedge.tri    = edgeNM.tri;
        sortEdges.push_back(sedge);
        nextNM = edgeNM.nextNM;
      }

      std::sort(sortEdges.data(), sortEdges.data() + sortEdges.size(), SortEdge::comparator);

      nextNM        = edge.nmList;
      edge.triRight = INVALID;

      for(uint32_t se = 0; se < sortEdges.size(); se++) {
        const SortEdge& sedge = sortEdges[se];

        if(!sedge.isLeft && edge.triRight == INVALID) {
          edge.triRight   = sedge.tri;
          edge.angleRight = sedge.angle;
        }
        else {
          EdgeNM& edgeNM = edgesNM[nextNM];
          edgeNM.angle   = sedge.angle;
          edgeNM.tri     = sedge.tri;
          edgeNM.isLeft  = sedge.isLeft;
          nextNM         = edgeNM.nextNM;
        }
      }
    }

    nonManifoldNeedsOrdering = false;
  }

  bool areTrianglesPaired(const Edge& edge, uint32_t t0, uint32_t t1) const
  {
    if((edge.triLeft == t0 && edge.triRight != t1) || (edge.triLeft == t1 && edge.triRight != t0))
      return false;

    if((edge.triLeft == t0 && edge.triRight == t1) || (edge.triLeft == t1 && edge.triRight == t0))
      return true;


    uint32_t triLeft  = INVALID;
    uint32_t triRight = INVALID;
    uint32_t nmList   = edge.nmList;

    // find next pairing
    while(nmList != INVALID) {
      const EdgeNM* edgeNM = iterateEdgeNM(nmList);
      if(edgeNM->isLeft) {
        triLeft  = edgeNM->tri;
        triRight = INVALID;
      }
      else {
        triRight = edgeNM->tri;
      }

      if(triLeft != INVALID && triRight != INVALID) {

        if((triLeft == t0 && triRight == t1) || (triLeft == t1 && triRight == t0))
          return true;

        triLeft  = INVALID;
        triRight = INVALID;
      }
    }

    return false;
  }

  bool iterateTrianglePairs(const Edge& edge, uint32_t& nmList, uint32_t& triLeft, uint32_t& triRight) const
  {
    if(triLeft == INVALID) {
      triLeft  = edge.triLeft;
      triRight = edge.triRight;
      nmList   = edge.nmList;
    }
    else {
      triLeft  = INVALID;
      triRight = INVALID;

      if(edge.isNonManifold()) {
        // find next pairing
        while(nmList != INVALID) {
          const EdgeNM* edgeNM = iterateEdgeNM(nmList);
          if(edgeNM->isLeft) {
            triLeft  = edgeNM->tri;
            triRight = INVALID;
          }
          else {
            triRight = edgeNM->tri;
          }

          assert(triLeft != triRight);

          if(triLeft != INVALID && triRight != INVALID)
            break;
        }
      }
    }

    return triLeft != INVALID && triRight != INVALID;
  }
};


//////////////////////////////////////////////////////////////////////////

static const uint32_t EDGE_HARD_BIT         = 1 << 0;
static const uint32_t EDGE_OPTIONAL_BIT     = 1 << 1;
static const uint32_t EDGE_HARD_FLOATER_BIT = 1 << 2;
static const uint32_t EDGE_MATERIAL_BIT     = 1 << 3;
static const uint32_t EDGE_ANGLE_BIT        = 1 << 4;
static const uint32_t EDGE_OVERLAP_BIT      = 1 << 5;

// separate class due to potential template usage for Mesh
class MeshUtils
{
public:
  static void removeCoplanarTriangle(Mesh& mesh, Mesh::Edge& edge, Loader::BuilderPart& builder, const LdrVector& normal, uint32_t t, uint32_t tOther)
  {
    if(!mesh.triAlive.getBit(t))
      return;

    if((builder.isSameTriangle(t, tOther) || builder.isSameQuad(t, tOther))) {
      mesh.removeTriangle(t);
      return;
    }

    LdrVector normalOther = mesh.getTriangleNormal(tOther, builder.positions.data());
    if(vec_dot(normal, normalOther) > Loader::COPLANAR_TRIANGLE_DOT) {
#if defined(_DEBUG) && LDR_DEBUG_PRINT_NON_MANIFOLDS
      printf("nonmanifold coplanar: t %d %d - %s\n", t, tOther, builder.filename.c_str());
#endif
      edge.flag |= EDGE_OVERLAP_BIT;
      mesh.hasOverlap = true;
    }
  }

  static void fillTriangles(Mesh& mesh, Loader::BuilderPart& builder)
  {
    uint32_t numT = builder.triangles.size() / 3;
    uint32_t numV = builder.positions.size();

    bool nonManifold = false;
    for(uint32_t t = 0; t < numT; t++) {
      uint32_t nonManifoldEdge = mesh.addTriangle(t);

      if(nonManifoldEdge != Mesh::INVALID) {
        Mesh::Edge& edge = mesh.edges[nonManifoldEdge];

        LdrVector normal = mesh.getTriangleNormal(t, builder.positions.data());

        if(edge.triLeft != t) {
          removeCoplanarTriangle(mesh, edge, builder, normal, t, edge.triLeft);
        }

        if(edge.triRight != Mesh::INVALID && edge.triRight != t) {
          removeCoplanarTriangle(mesh, edge, builder, normal, t, edge.triRight);
        }

        if(edge.isNonManifold()) {
          uint32_t numLeft = 0;
          uint32_t num     = 0;
          uint32_t nextNM  = edge.nmList;
          while(nextNM != Mesh::INVALID) {
            uint32_t tOther = mesh.iterateEdgeNM(nextNM)->tri;
            num++;
            if(tOther != t) {
              removeCoplanarTriangle(mesh, edge, builder, normal, t, tOther);
            }
          }

#if defined(_DEBUG) && LDR_DEBUG_PRINT_NON_MANIFOLDS
          if(num > 3)
            printf("nonmanifold high edge side: t %d - %d - %s\n", t, num, builder.filename.c_str());
#endif
        }
      }
      nonManifold = nonManifold || (nonManifoldEdge != Mesh::INVALID);
    }
    mesh.nonManifoldEdges = nonManifold;
  }

  static void removeDeleted(Mesh& mesh, Loader::BuilderPart& builder)
  {
    Loader::TVector<uint32_t> remap(mesh.numTriangles);

    uint32_t write = 0;
    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      if(mesh.triAlive.getBit(t)) {
        builder.triangles[write * 3 + 0] = builder.triangles[t * 3 + 0];
        builder.triangles[write * 3 + 1] = builder.triangles[t * 3 + 1];
        builder.triangles[write * 3 + 2] = builder.triangles[t * 3 + 2];
        builder.triangleMaterials[write] = builder.triangleMaterials[t];
        remap[t]                         = write;
        LdrNgon ngon                     = builder.triangleNgons[t];
        ngon.index                       = remap[ngon.index];
        builder.triangleNgons[write]     = ngon;
        write++;
      }
    }

    builder.triangles.resize(write * 3);
    builder.triangleMaterials.resize(write);
    builder.triangleNgons.resize(write);
  }

  static void storeEdgeLines(Mesh& mesh, Loader::BuilderPart& builder, bool optional)
  {
    Loader::TVector<LdrVertexIndex>& lines = optional ? builder.optional_lines : builder.lines;
    uint32_t                         flag  = optional ? EDGE_OPTIONAL_BIT : EDGE_HARD_BIT;

    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      const Mesh::Edge& edge = mesh.edges[e];
      if(!edge.isDead() && edge.flag & flag) {
        lines.push_back(edge.vtxA);
        lines.push_back(edge.vtxB);
      }
    }
  }
  static void fillLines(Mesh& mesh, Loader::BuilderPart& builder, bool optional)
  {
    Loader::TVector<uint32_t> path;
    path.reserve(16);

    Loader::TVector<LdrVertexIndex>& lines = optional ? builder.optional_lines : builder.lines;
    uint32_t                         flag  = optional ? EDGE_OPTIONAL_BIT : EDGE_HARD_BIT;

    // flag mesh edges as lines
    // Some fixing required due to floaters (lines are not actually existing as triangle edges)
    // or due to non-manifold fix before.

    uint32_t numOrigLines = lines.size() / 2;

    Loader::TVector<LdrVertexIndex> newLines;

    for(uint32_t i = 0; i < lines.size() / 2; i++) {
      uint32_t lineA = lines[i * 2 + 0];
      uint32_t lineB = lines[i * 2 + 1];

      Mesh::Edge* edgeFound = mesh.getEdge(lineA, lineB);
      if(edgeFound) {
        edgeFound->flag |= flag;
      }
      else {
        // floating edges (no triangles) are ugly, try to find path between them
        // once from both directions in case there is t-junction

        uint32_t foundConnections = 0;

        for(int side = 0; side < 2; side++) {
          uint32_t vStart = side ? lineA : lineB;
          uint32_t vEnd   = side ? lineB : lineA;

          LdrVector posEnd  = builder.positions[vEnd];
          LdrVector vecEdge = vec_normalize(vec_sub(posEnd, builder.positions[vStart]));

          path.clear();

          float minDist = FLT_MAX;

          uint32_t v       = vStart;
          uint32_t numPath = 0;

          while(v != vEnd && v != LDR_INVALID_IDX) {
            float    maxDot      = 0.90f;
            uint32_t closestEdge = LDR_INVALID_IDX;
            uint32_t vNext       = LDR_INVALID_IDX;

            // find closest
            uint32_t vtests[1] = {v};
            for(uint32_t t = 0; t < 1; t++) {
              uint32_t vtest = vtests[t];
              if(vtest == LDR_INVALID_IDX)
                continue;

              uint32_t        edgeCount;
              const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(vtest, edgeCount);
              for(uint32_t e = 0; e < edgeCount; e++) {
                const Mesh::Edge& edge = mesh.edges[edgeIndices[e]];
                assert(!edge.isDead());
                uint32_t  vOther = edge.otherVertex(vtest);
                LdrVector vecCur = vec_normalize(vec_sub(builder.positions[vOther], builder.positions[vtest]));
                float     dist   = vec_sq_length(vec_sub(builder.positions[vOther], builder.positions[vEnd]));
                float     dot    = vec_dot(vecCur, vecEdge);
                if(dot > maxDot && dist < minDist) {
                  maxDot      = dot;
                  vNext       = vOther;
                  closestEdge = edgeIndices[e];
                }
              }
            };

            v = vNext;
            path.push_back(closestEdge);
            numPath++;
          }

          if(v == vEnd) {
            for(uint32_t p = 0; p < numPath; p++) {
              mesh.edges[path[p]].flag |= flag;
            }
            foundConnections++;
          }
        }

        if(foundConnections != 2) {
          newLines.push_back(lineA);
          newLines.push_back(lineB);
        }
      }
    }
    //lines.move(newLines);
    lines = std::move(newLines);
  }

  struct CoplanarTriangleRegions
  {
    Loader::TVector<float>     trianglesArea;
    Loader::TVector<float>     trianglesConnectedArea;
    Loader::TVector<uint32_t>  trianglesConnected;
    Loader::TVector<LdrVector> trianglesNormals;
    Loader::BitArray           trianglesVisited;

    Loader::TVector<uint32_t> tempQueue;
    uint32_t                  readPos  = 0;
    uint32_t                  writePos = 0;

    CoplanarTriangleRegions(Mesh& mesh, Loader::BuilderPart& builder)
    {
      trianglesArea.resize(mesh.numTriangles, 0.0f);
      trianglesConnectedArea.resize(mesh.numTriangles, 0.0f);
      trianglesConnected.resize(mesh.numTriangles);
      trianglesNormals.resize(mesh.numTriangles);
      trianglesVisited.resize(mesh.numTriangles, false);

      tempQueue.resize(mesh.numTriangles + 1);

      for(uint32_t t = 0; t < mesh.numTriangles; t++) {
        trianglesConnected[t] = t;
      }
      for(uint32_t t = 0; t < mesh.numTriangles; t++) {
        float area;
        trianglesNormals[t] = vec_normalize_length(mesh.getTriangleCross(t, builder.positions.data()), area);
        trianglesArea[t]    = 0.5f * area;
      }
    }

    void buildRegion(Mesh& mesh, Loader::BuilderPart& builder, uint32_t tSeed)
    {
      tempQueue[0] = tSeed;
      readPos      = 0;
      writePos     = 1;

      trianglesVisited.setBit(tSeed, true);

      while(readPos != writePos) {
        uint32_t t = tempQueue[readPos];

        trianglesConnected[t] = tSeed;
        trianglesConnectedArea[tSeed] += trianglesArea[t];

        for(uint32_t e = 0; e < 3; e++) {
          Mesh::VertexPair  pair   = mesh.getTriangleEdgeVertices(t, e);
          const Mesh::Edge* edge   = mesh.getEdge(pair.a, pair.b);
          uint32_t          tOther = edge->otherTri(t);
          if(edge->isNonManifold() || edge->isOpen() || trianglesVisited.getBit(tOther)
             || vec_dot(trianglesNormals[t], trianglesNormals[tOther]) <= Loader::COPLANAR_TRIANGLE_DOT) {
            continue;
          }

          trianglesVisited.setBit(tOther, true);
          tempQueue[writePos++] = tOther;
        }

        readPos++;
      }
    }
  };

  static void fixRegionOverlap(Mesh& mesh, Loader::BuilderPart& builder)
  {
    CoplanarTriangleRegions regions(mesh, builder);
    Loader::BitArray        regionsDelete(mesh.numTriangles, false);

    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      const Mesh::Edge& edge = mesh.edges[e];
      if(!(edge.flag & EDGE_OVERLAP_BIT))
        continue;

      // find coplanar
      uint32_t nextNM = edge.nmList;
      while(nextNM != Mesh::INVALID) {
        const Mesh::EdgeNM* edgeNM = mesh.iterateEdgeNM(nextNM);
        uint32_t            tA     = edgeNM->isLeft ? edge.triLeft : edge.triRight;
        uint32_t            tB     = edgeNM->tri;

        if(tA != LDR_INVALID_IDX && vec_dot(regions.trianglesNormals[tA], regions.trianglesNormals[tB]) > Loader::COPLANAR_TRIANGLE_DOT) {
          // build connected region for tA and tB
          if(!regions.trianglesVisited.getBit(tA)) {
            regions.buildRegion(mesh, builder, tA);
          }
          if(!regions.trianglesVisited.getBit(tB)) {
            regions.buildRegion(mesh, builder, tB);
          }
          if(regions.trianglesConnected[tA] != regions.trianglesConnected[tB]) {
            // compare area
            uint32_t tRegion = regions.trianglesConnectedArea[regions.trianglesConnected[tA]]
                                       > regions.trianglesConnectedArea[regions.trianglesConnected[tB]] ?
                                   tB :
                                   tA;
            regionsDelete.setBit(tRegion, true);
          }
        }
      }
    }

    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      if(regionsDelete.getBit(regions.trianglesConnected[t])) {
        mesh.removeTriangle(t);
      }
    }

    mesh.hasOverlap = false;
  }

  struct NgonTriangulation
  {
    // current algorithm from here
    // https://www.flipcode.com/archives/Efficient_Polygon_Triangulation.shtml
    //
    // TODO need something that prefers shorter edge lengths
    // try Triangulate_OPT from https://github.com/ivanfratric/polypartition


    uint32_t                   numOutlineVertices = 0;
    Loader::TVector<uint32_t>  workingSet;
    Loader::TVector<LdrVector> outlineVertices;
    Loader::TVector<uint32_t>  triangleVertices;
    LdrVector                  basis[3];

    inline bool isInsideTriangle(LdrVector a, LdrVector b, LdrVector c, LdrVector p)
    {
      LdrVector ab = vec_sub(b, a);
      LdrVector bc = vec_sub(c, b);
      LdrVector ca = vec_sub(a, c);

      LdrVector ap = vec_sub(p, a);
      LdrVector bp = vec_sub(p, b);
      LdrVector cp = vec_sub(p, c);

      float pCrossAB = ab.x * ap.y - ab.y * ap.x;
      float pCrossBC = bc.x * bp.y - bc.y * bp.x;
      float pCrossCA = ca.x * cp.y - ca.y * cp.x;

      return ((pCrossAB >= 0.0f) && (pCrossBC >= 0.0f) && (pCrossCA >= 0.0f));
    }

    inline LdrVector makeOutlineVertex(const Loader::BuilderPart& builder, uint32_t i)
    {
      LdrVector pos = builder.positions[i];
      LdrVector projected;
      projected.x = vec_dot(basis[0], pos);
      projected.y = vec_dot(basis[1], pos);
      projected.z = 0.0f;

      return projected;
    }

    inline bool snip(const Loader::BuilderPart& builder, int32_t u, int32_t v, int32_t w, int32_t numPoints)
    {
      LdrVector a = outlineVertices[u];
      LdrVector b = outlineVertices[v];
      LdrVector c = outlineVertices[w];

      if(0.0000000001f > (((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x))))
        return false;

      for(int32_t op = 0; op < numPoints; op++) {
        if((op == u) || (op == v) || (op == w))
          continue;

        LdrVector p = outlineVertices[op];

        if(isInsideTriangle(a, b, c, p))
          return false;
      }

      return true;
    }

    void build(Mesh& mesh, Loader::BuilderPart& builder, uint32_t triSeed, const uint32_t* triangleLinkedList)
    {
      uint32_t numTriangles = builder.triangleNgons[triSeed].num;


      workingSet.clear();
      workingSet.reserve(numTriangles + 4);
      triangleVertices.clear();
      triangleVertices.reserve(numTriangles * 3);

      mesh.getTriangleBasis(triSeed, builder.positions.data(), basis);

      // build initial outline
      uint32_t triLast = triSeed;
      uint32_t tri     = triangleLinkedList[triSeed];
      uint32_t vtxLast = LDR_INVALID_IDX;

      for(uint32_t t = 0; t < numTriangles; t++) {
        assert(tri != LDR_INVALID_IDX);

        uint32_t connectedVertices[3];
        uint32_t connectedVerticesLast[3];
        uint32_t numConnected = mesh.findTriangleConnectingVertices(tri, triLast, connectedVertices, connectedVerticesLast);

        uint32_t vtxJoin    = LDR_INVALID_IDX;
        uint32_t vtxPreJoin = LDR_INVALID_IDX;

        if(numConnected == 2) {
          // the two triangles are connected like a fan
          // determine which of the 2 vertices starts the edge that leads to the "new" vertex
          // that isn't connected.
          // We walk those "outer" edges of the ngon first.
          uint32_t mask = 7;
          mask &= ~((1 << connectedVertices[0]) | (1 << connectedVertices[1]));
          // mask is 1,2,4 -> 0,1,2
          uint32_t newTriangleVertex = mask / 2;
          uint32_t connectedVertex   = (newTriangleVertex + 3 - 1) % 3;
          bool     isFirst           = connectedVertex == connectedVertices[0];
          assert(isFirst || connectedVertex == connectedVertices[1]);

          connectedVertices[0]     = connectedVertex;
          connectedVerticesLast[0] = isFirst ? connectedVerticesLast[0] : connectedVerticesLast[1];

          numConnected = 1;
        }

        if(numConnected == 1) {
          vtxJoin = mesh.getTriangle(tri)[connectedVertices[0]];
          // pick start vertex of edge that ends with join vertex
          vtxPreJoin = mesh.getTriangle(triLast)[(connectedVerticesLast[0] + 3 - 1) % 3];
        }
        else {
          assert(0 && "invalid number of connected vertices between ngon triangles");
        }

        if(vtxPreJoin != LDR_INVALID_IDX && vtxPreJoin != vtxLast) {
          workingSet.push_back(vtxPreJoin);
        }
        if(vtxJoin != LDR_INVALID_IDX) {
          workingSet.push_back(vtxJoin);
          vtxLast = vtxJoin;
        }

        triLast = tri;
        tri     = triangleLinkedList[tri];
      }
      // list must come around to seed
      assert(triLast == triSeed);

      numOutlineVertices = uint32_t(workingSet.size());

      outlineVertices.resize(workingSet.size());

      for(uint32_t v = 0; v < numOutlineVertices; v++) {
        outlineVertices[v] = makeOutlineVertex(builder, workingSet[v]);
      }

      // build new triangles
      {
        int32_t nv    = int32_t(numOutlineVertices);
        int32_t count = int32_t(2 * numOutlineVertices);

        for(int32_t m = 0, v = nv - 1; nv > 2;) {
          if(0 >= (count--)) {
            return;
          }

          int32_t u = v;
          if(nv <= u) {
            u = 0;
          }
          v = u + 1;
          if(nv <= v) {
            v = 0;
          }
          int32_t w = v + 1;
          if(nv <= w) {
            w = 0;
          }
          if(snip(builder, u, v, w, nv)) {
            uint32_t a = workingSet[u];
            uint32_t b = workingSet[v];
            uint32_t c = workingSet[w];

            triangleVertices.push_back(a);
            triangleVertices.push_back(b);
            triangleVertices.push_back(c);

            m++;

            int32_t s;
            int32_t t;
            for(s = v, t = v + 1; t < nv; s++, t++) {
              workingSet[s] = workingSet[t];
            }
            nv--;

            count = 2 * nv;
          }
        }
      }

      // replace triangles of linked list with new ones
      if(triangleVertices.size() == numTriangles * 3) {
        // remove all with old triangulation
        tri = triSeed;
        for(uint32_t t = 0; t < numTriangles; t++) {
          assert(tri != LDR_INVALID_IDX);

          // don't flag dead given we will re-add with new topology
          mesh.removeTriangle(tri);

          tri = triangleLinkedList[tri];
        }
        assert(tri == triSeed || tri == LDR_INVALID_IDX);

        // add all with new triangulation
        tri = triSeed;
        for(uint32_t t = 0; t < numTriangles; t++) {
          assert(tri != LDR_INVALID_IDX);

          // replace with new indices
          builder.triangles[tri * 3 + 0] = triangleVertices[t * 3 + 0];
          builder.triangles[tri * 3 + 1] = triangleVertices[t * 3 + 1];
          builder.triangles[tri * 3 + 2] = triangleVertices[t * 3 + 2];

          // add back
          mesh.addTriangle(tri);

          tri = triangleLinkedList[tri];
        }
      }
    }
  };

  static bool fixTjunctions(Mesh& mesh, Loader::BuilderPart& builder)
  {
    bool modified = false;

    // removes t-junctions and closable gaps

    uint32_t numVertices = (uint32_t)builder.positions.size();

    Loader::BitArray processed(mesh.edges.size(), false);

    Loader::TVector<uint32_t> triangleNgonList(mesh.numTriangles, LDR_INVALID_IDX);

    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      // prior fixing t-junctions only tris and quads exist
      // the second triangle will setup the circular list
      LdrNgon ngon = builder.triangleNgons[t];
      if(ngon.index != t) {
        triangleNgonList[t]          = ngon.index;
        triangleNgonList[ngon.index] = t;
      }
      else if(ngon.num == 1) {
        triangleNgonList[t] = t;
      }
    }

    for(uint32_t v = 0; v < numVertices; v++) {
      bool     respin = false;
      uint32_t edgeCountA;
      mesh.vtxEdges.getConnected(v, edgeCountA);

      bool isOpenCorner = mesh.getVertexClosable(v) && edgeCountA == 2;

      for(uint32_t ea = 0; ea < edgeCountA; ea++) {
        // must get pointer inside loop, given we modify the mesh when creating new triangles
        const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(v, edgeCountA);

        uint32_t   edgeIdxA = edgeIndices[ea];
        Mesh::Edge edgeA    = mesh.edges[edgeIdxA];

        // only edges that start from this vertex (so we have canonical winding)
        // don't want to start from vertex whose edge is "reversed"
        if(processed.getBit(edgeIdxA) || edgeA.vtxA != v || !edgeA.isOpen() || edgeA.isDead())
          continue;

        uint32_t triOld = edgeA.triLeft;

        LdrVector posA = builder.positions[v];
        uint32_t  vEnd = edgeA.otherVertex(v);

        float     lengthA;
        LdrVector vecA = vec_normalize_length(vec_sub(builder.positions[vEnd], posA), lengthA);

        std::vector<uint32_t> path;
        path.reserve(128);

        uint32_t vNext          = v;
        uint32_t edgeIdxSkip    = edgeIdxA;
        uint32_t edgeNext       = 0;
        bool     singleTriangle = true;

        // find closest edges
        while(edgeNext != Mesh::INVALID) {
          uint32_t        edgeCountC;
          const uint32_t* edgeIndicesC = mesh.vtxEdges.getConnected(vNext, edgeCountC);
          float           maxDot       = 0.98f;
          uint32_t        idx          = Mesh::INVALID;

          for(uint32_t ec = 0; ec < edgeCountC; ec++) {
            uint32_t          edgeIdxC = edgeIndicesC[ec];
            const Mesh::Edge& edgeC    = mesh.edges[edgeIndicesC[ec]];

            if(edgeIdxC == edgeIdxSkip || edgeC.isDead() || !edgeC.isOpen() || processed.getBit(edgeIdxC))
              continue;

            uint32_t vC = edgeC.otherVertex(vNext);

            float     lengthC;
            LdrVector vecC  = vec_normalize_length(vec_sub(builder.positions[vC], posA), lengthC);
            float     dotAC = vec_dot(vecA, vecC);
            if((lengthC <= lengthA * 1.05f && dotAC > maxDot) || vC == vEnd) {
              maxDot = dotAC;
              idx    = edgeIdxC;
            }
          }
          edgeNext    = idx;
          edgeIdxSkip = edgeNext;

          if(edgeNext != Mesh::INVALID) {
            path.push_back(edgeNext);
            vNext = mesh.edges[edgeNext].otherVertex(vNext);

            singleTriangle = singleTriangle && mesh.edges[edgeNext].triLeft == triOld;

            if(vNext == vEnd || path.size() == 128) {
              break;
            }
          }
        };

        if(vNext != vEnd)
          continue;

        // special case close triangle holes
        if(isOpenCorner) {
          uint32_t vPrev = mesh.edges[edgeNext].otherVertex(vNext);
          if(mesh.getVertexClosable(vPrev)) {
            uint32_t triOld = edgeA.triLeft;
            mesh.removeTriangle(triOld);
            mesh.replaceTriangleVertex(triOld, v, vPrev);
            mesh.addTriangle(triOld);
            ea = edgeCountA;

            modified = true;

            processed.resize(mesh.edges.size(), false);
            continue;
          }
        }
        if(singleTriangle) {
          continue;
        }

        modified = true;

        processed.setBit(edgeIdxA, true);

        uint32_t triIndices[3];
        triIndices[0] = builder.triangles[triOld * 3 + 0];
        triIndices[1] = builder.triangles[triOld * 3 + 1];
        triIndices[2] = builder.triangles[triOld * 3 + 2];

        uint32_t subEdge = 0;
        uint32_t vCorner = 0;
        if(triIndices[0] == edgeA.vtxA) {
          subEdge = 0;
          vCorner = triIndices[2];
        }
        else if(triIndices[1] == edgeA.vtxA) {
          subEdge = 1;
          vCorner = triIndices[0];
        }
        else if(triIndices[2] == edgeA.vtxA) {
          subEdge = 2;
          vCorner = triIndices[1];
        }
        else {
          assert(0);
        }

        // preserve edge flags from triangle that we rebuild (edge could be deleted during removal)
        uint32_t flagACorner = mesh.getEdge(edgeA.vtxA, vCorner)->flag;
        uint32_t flagBCorner = mesh.getEdge(edgeA.vtxB, vCorner)->flag;

        mesh.removeTriangle(triOld);

        if(triOld == 454 || triOld == 455)
          triOld = triOld;

        LdrNgon ngon = builder.triangleNgons[triOld];
        ngon         = builder.triangleNgons[ngon.index];

        // append newly added
        ngon.num += uint32_t(path.size()) - 1;
        builder.triangleNgons[ngon.index] = ngon;
        builder.triangleNgons[triOld]     = ngon;

        uint32_t flagPath = edgeA.flag;
        uint32_t vFirst   = edgeA.vtxA;

        uint32_t lastTri = triOld;

        for(size_t i = 0; i < path.size(); i++) {
          uint32_t    edgeIdxC = path[i];
          Mesh::Edge& edgeC    = mesh.edges[edgeIdxC];
          processed.setBit(edgeIdxC, true);
          // inherit edge flag
          edgeC.flag |= flagPath;

          // make new triangles, first re-uses old slot

          uint32_t triIdx;
          if(i != 0) {
            triIdx = uint32_t(builder.triangles.size() / 3);
            builder.triangles.push_back(vFirst);
            builder.triangles.push_back(edgeC.otherVertex(vFirst));
            builder.triangles.push_back(vCorner);
            builder.triangleMaterials.push_back(builder.triangleMaterials[triOld]);
            builder.triangleNgons.push_back(ngon);

            // insert into ngon list
            triangleNgonList.push_back(triangleNgonList[lastTri]);
            triangleNgonList[lastTri] = triIdx;

            lastTri = triIdx;
          }
          else {
            triIdx                            = triOld;
            builder.triangles[triOld * 3 + 0] = vFirst;
            builder.triangles[triOld * 3 + 1] = edgeC.otherVertex(vFirst);
            builder.triangles[triOld * 3 + 2] = vCorner;
          }

          mesh.triangles = builder.triangles.data();

          uint32_t nonManifold = mesh.addTriangle(triIdx);
          vFirst               = edgeC.otherVertex(vFirst);
        }

        // re-apply edge flag
        Mesh::Edge* edgeACorner = mesh.getEdge(edgeA.vtxA, vCorner);
        Mesh::Edge* edgeBCorner = mesh.getEdge(edgeA.vtxB, vCorner);
        if(edgeACorner)
          edgeACorner->flag = flagACorner;
        if(edgeBCorner)
          edgeBCorner->flag = flagBCorner;

        processed.resize(mesh.edges.size(), false);
        respin = true;
      }
      if(respin) {
        v = v - 1;
      }
    }

#if 0
    NgonTriangulation ngonTriangulation;
    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      LdrNgon ngon = builder.triangleNgons[t];
      if(ngon.num <= 2 || ngon.index != t)
        continue;
      // rebuild triangulation of ngons

      ngonTriangulation.build(mesh, builder, t, triangleNgonList.data());
    }
#endif
    return modified;
  }

  static void fixBuilderPart(Loader::BuilderPart& builder, const Loader::Config& config)
  {
    Mesh mesh;
    mesh.initBasics(builder.positions.size(), builder.triangles.size() / 3, builder.triangles.data(), builder.loader);

    builder.flag.canChamfer = 1;

    MeshUtils::fillTriangles(mesh, builder);

    MeshUtils::fillLines(mesh, builder, false);
    MeshUtils::fillLines(mesh, builder, true);
    if(config.partFixOverlap && mesh.hasOverlap) {
      MeshUtils::fixRegionOverlap(mesh, builder);
    }
    if(config.partFixTjunctions) {
      MeshUtils::fixTjunctions(mesh, builder);
    }
    MeshUtils::storeEdgeLines(mesh, builder, false);
    MeshUtils::storeEdgeLines(mesh, builder, true);
    MeshUtils::removeDeleted(mesh, builder);
  }

  static void chamferRenderPart(Mesh& mesh, Loader::BuilderRenderPart& builder, const LdrPart& part, const float chamferPreferred)
  {
    assert(mesh.requiresVertexTriangles);

    // copy over original triangles first
    Loader::TVector<uint32_t> renderTriangles = builder.triangles;
    builder.trianglesC                        = builder.triangles;
    builder.materialsC.resize(part.flag.hasComplexMaterial ? part.numTriangles : 0);

    bool hasMaterials = !builder.materialsC.empty();

    if(hasMaterials) {
      memcpy(builder.materialsC.data(), part.triangleMaterials, sizeof(LdrMaterialID) * part.numTriangles);
    }

    // all vertices that were split before to account for hard edges are relevant here
    // that means we don't chamfer open-edges

    // first pass is to create new chamfer vertices
    // one vertex per every split vertex

    // store where the chamfered begin
    Loader::TVector<uint32_t> vtxChamferBegin(part.numPositions, 0);

    for(uint32_t v = 0; v < part.numPositions; v++) {
      Loader::BuilderRenderPart::VertexInfo outInfo  = builder.vtxOutInfo[v];
      uint32_t                              outCount = outInfo.count;
      if(outCount > 1) {

        vtxChamferBegin[v] = (uint32_t)builder.vertices.size();

        // find minimum chamfer length (distances per triangle cluster would be safer)
        uint32_t        edgeCount;
        const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(v, edgeCount);

        float dist = FLT_MAX;

        LdrVector avgNormal = {0, 0, 0};
        LdrVector avgPos    = {0, 0, 0};

        for(uint32_t e = 0; e < edgeCount; e++) {
          const Mesh::Edge edge = mesh.edges[edgeIndices[e]];
          dist = std::min(dist, vec_sq_length(vec_sub(part.positions[edge.vtxA], part.positions[edge.vtxB])));
        }
        const float chamferDistance = std::min(chamferPreferred, dist * 0.40f);

        uint32_t materialTri;

        // create new output vertex that is chamfered
        for(uint32_t o = 0; o < outCount; o++) {
          uint32_t outIdx = builder.vtxOutIndices[outInfo.begin + o];

          LdrVector normal = builder.vertices[outIdx].normal;

          const Loader::BuilderRenderPart::EdgePair& edgePair = builder.vtxOutEdgePairs[outInfo.begin + o];
          // get the two edges that define the triangle cluster for this output vertex
          const Mesh::Edge& edgeA = mesh.edges[edgePair.edge[0]];
          const Mesh::Edge& edgeB = mesh.edges[edgePair.edge[1]];
          uint32_t          tri   = edgePair.tri[0];
          materialTri             = tri;

          // algorithm described by Diana Algma https://github.com/dianx93
          // https://comserv.cs.ut.ee/home/files/Algma_ComputerScience_2018.pdf?study=ATILoputoo&reference=D4FE5BC8A22718CF3A52B308AD2B2B878C78EB36


          //  aligned
          //
          //  |     |     |
          //  C <-- B --> A
          //
          //  unaligned
          //
          //  shift = -avg
          //
          //  |      /
          //  C <-- B
          //         \  /
          //          A
          //
          //  or shift = avg
          //
          //           \
          //            A
          //  |    \  /
          //  C <-- B
          //

          LdrVector vecBC = vec_sub(part.positions[edgeA.otherVertex(v)], part.positions[v]);
          LdrVector vecBA = vec_sub(part.positions[edgeB.otherVertex(v)], part.positions[v]);

          // may need to reverse vectors to makes sure edge BC is "against" triangle winding.
          if(!((edgeA.vtxB == v && edgeA.triLeft == edgePair.tri[0]) || (edgeA.vtxA == v && edgeA.triRight == edgePair.tri[0]))) {
            LdrVector temp = vecBC;
            vecBC          = vecBA;
            vecBA          = temp;
          }

          // project into normal plane
          vecBA = vec_sub(vecBA, vec_mul(normal, vec_dot(normal, vecBA)));
          vecBC = vec_sub(vecBC, vec_mul(normal, vec_dot(normal, vecBC)));

          vecBA = vec_normalize(vecBA);
          vecBC = vec_normalize(vecBC);

          float chamferModifier = 1.0f;
          bool  isOpposite      = vec_dot(vecBA, vecBC) < -Loader::CHAMFER_PARALLEL_DOT;
          bool  isSame          = vec_dot(vecBA, vecBC) > Loader::CHAMFER_PARALLEL_DOT;

          LdrVector shift;
          if(isSame) {
            chamferModifier = 0;
            shift           = {0, 0, 0};
          }
          else if(isOpposite) {
            shift = vec_normalize(vec_cross(vecBC, normal));
          }
          else {
            shift = vec_normalize(vec_add(vecBA, vecBC));
            // may need to flip side
            float chamferSign = vec_dot(vec_cross(vecBA, vecBC), normal) < 0 ? -1.0f : 1.0f;

            float h = 1.0f / sinf(acosf(vec_dot(vecBA, vecBC)) * 0.5f);
            if(h > 10.0f) {
              h = 0;
            }

            chamferModifier *= chamferSign * h;
          }

          // just in case things go beyond south
          if(isnan(chamferModifier) || isinf(chamferModifier) || (edgeA.isOpen() && edgeB.isOpen())) {
            chamferModifier = 0;
          }

          LdrVector vecDelta = vec_mul(shift, (chamferDistance * chamferModifier));

          if(edgeA.isOpen() || edgeB.isOpen()) {
            uint32_t  vOther  = edgeA.isOpen() ? edgeA.otherVertex(v) : edgeB.otherVertex(v);
            LdrVector vecOpen = vec_normalize(vec_sub(part.positions[vOther], part.positions[v]));
            vecDelta          = vec_mul(vecOpen, vec_dot(vecOpen, vecDelta));
          }

          uint32_t        newIdx    = (uint32_t)builder.vertices.size();
          LdrRenderVertex newVertex = builder.vertices[outIdx];
          newVertex.position        = vec_add(newVertex.position, vecDelta);
          builder.vertices.push_back(newVertex);

          avgNormal = vec_add(avgNormal, newVertex.normal);
          avgPos    = vec_add(avgPos, newVertex.position);

          // replace vertices with shifted version in adjacent triangles

          uint32_t        triCount;
          const uint32_t* triIndices = mesh.vtxTriangles.getConnected(v, triCount);

          for(uint32_t t = 0; t < triCount; t++) {
            const uint32_t  triBegin = triIndices[t] * 3;
            const uint32_t* indices  = &renderTriangles[triBegin];
            if(indices[0] == outIdx)
              builder.trianglesC[triBegin + 0] = newIdx;
            if(indices[1] == outIdx)
              builder.trianglesC[triBegin + 1] = newIdx;
            if(indices[2] == outIdx)
              builder.trianglesC[triBegin + 2] = newIdx;
          }
        }

        uint32_t groupStart  = 0;
        uint32_t groupLength = 0;
        for(uint32_t g = 0; g < outInfo.groupCount; g++) {

          groupLength = 0;
          for(uint32_t og = groupStart; og < outInfo.count; og++) {
            const Loader::BuilderRenderPart::EdgePair& pair = builder.vtxOutEdgePairs[outInfo.begin + og];
            if(pair.group != g) {
              break;
            }
            groupLength++;
          }

          // create "inner triangle" based on out vertex count
          // 2 -> nothing
          // 3 -> 1 triangle
          // 4 -> 2 triangles
          // 5 -> 5 triangles with center point (NYI)

          // fixme, naive actual implementation outcount > 4 needs other logic

          uint32_t outTriangleCount = groupLength > 4 ? groupLength : (groupLength > 2 ? groupLength - 2 : 0);
          uint32_t lastIndex        = builder.vertices.size();
          if(outTriangleCount > 2) {
            avgNormal = vec_normalize(avgNormal);
            builder.vertices.push_back({vec_mul(avgPos, 1.0f / float(groupLength)), 0, avgNormal});
          }

          for(uint32_t t = 0; t < outTriangleCount; t++) {
            uint32_t a = vtxChamferBegin[v] + groupStart + t + 0;
            uint32_t b = vtxChamferBegin[v] + groupStart + (t + 1) % groupLength;
            uint32_t c = outTriangleCount > 2 ? lastIndex : vtxChamferBegin[v] + groupStart + t + 2;

            LdrVector normal = vec_cross(vec_sub(builder.vertices[b].position, builder.vertices[a].position),
                                         vec_sub(builder.vertices[c].position, builder.vertices[a].position));

            if(vec_dot(normal, avgNormal) < 0) {
              builder.trianglesC.push_back(c);
              builder.trianglesC.push_back(b);
              builder.trianglesC.push_back(a);
            }
            else {
              builder.trianglesC.push_back(a);
              builder.trianglesC.push_back(b);
              builder.trianglesC.push_back(c);
            }

            if(hasMaterials) {
              builder.materialsC.push_back(part.triangleMaterials[materialTri]);
            }
          }

          groupStart += groupLength;
        }
      }
    }

    // second pass is to create new triangles for every hard edge
    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      const Mesh::Edge& edge = mesh.edges[e];

      if(edge.isOpen() || !(edge.flag & (EDGE_ANGLE_BIT | EDGE_HARD_BIT)))
        continue;

      uint32_t triLeft  = Mesh::INVALID;
      uint32_t triRight = Mesh::INVALID;
      uint32_t nmList   = Mesh::INVALID;

      while(mesh.iterateTrianglePairs(edge, nmList, triLeft, triRight)) {

        // split edge exists if
        // - we have an edge between the two original triangles
        // - but they are not connected in the rendermesh

        if(mesh.areTriSmoothAdjacent(triLeft, triRight))
          continue;

        // find the right out vertex for this edge
        auto findChamferVertex = [&](uint32_t v, uint32_t tri) {
          Loader::BuilderRenderPart::VertexInfo outInfo  = builder.vtxOutInfo[v];
          uint32_t                              outCount = outInfo.count;
          for(uint32_t o = 0; o < outCount; o++) {
            uint32_t                                   outIdx   = builder.vtxOutIndices[outInfo.begin + o];
            const Loader::BuilderRenderPart::EdgePair& edgePair = builder.vtxOutEdgePairs[outInfo.begin + o];
            if(edgePair.edge[0] == e && edgePair.tri[0] == tri) {
              return vtxChamferBegin[v] + o;
            }
            if(edgePair.edge[1] == e && edgePair.tri[1] == tri) {
              return vtxChamferBegin[v] + o;
            }
          }
          assert(outCount == 1);
          return builder.vtxOutIndices[outInfo.begin];
        };

        // find the two vertices for left side
        uint32_t idxLeftA = findChamferVertex(edge.vtxA, triLeft);
        uint32_t idxLeftB = findChamferVertex(edge.vtxB, triLeft);
        // for right side
        uint32_t idxRightA = findChamferVertex(edge.vtxA, triRight);
        uint32_t idxRightB = findChamferVertex(edge.vtxB, triRight);

        uint32_t triangles[2][3] = {
            {idxLeftA, idxRightB, idxLeftB},
            {idxRightA, idxRightB, idxLeftA},
        };

        for(uint32_t t = 0; t < 2; t++) {
          uint32_t a = triangles[t][0];
          uint32_t b = triangles[t][1];
          uint32_t c = triangles[t][2];

          if(a == b || a == c || b == c)
            continue;

          if(hasMaterials) {
            builder.materialsC.push_back(part.triangleMaterials[triLeft]);
          }
          builder.trianglesC.push_back(a);
          builder.trianglesC.push_back(b);
          builder.trianglesC.push_back(c);
        }
      }
    }
  }

  static LdrRenderVertex make_vertex(const LdrVector& pos, const LdrVector& inNormal)
  {
    LdrRenderVertex vertex;
    vertex.position = pos;
    vertex.normal   = vec_normalize(inNormal);

    return vertex;
  }

  static LdrRenderVertex make_vertex(const LdrVector& pos)
  {
    LdrRenderVertex vertex;
    vertex.position = pos;
    vertex.normal   = vec_normalize(pos);

    return vertex;
  }

  static void buildRenderPartBasic(Loader::BuilderRenderPart& builder, Mesh& mesh, const LdrPart& part, const Loader::Config& config)
  {
    assert(mesh.requiresVertexTriangles);

    // start with unique vertices per triangle

    builder.vertices.resize(mesh.numTriangles * 3);

    uint32_t* triangleVertices = builder.triangles.data();

    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      for(uint32_t k = 0; k < 3; k++) {
        triangleVertices[t * 3 + k]          = t * 3 + k;
        builder.vertices[t * 3 + k].position = part.positions[mesh.triangles[t * 3 + k]];
        builder.vertices[t * 3 + k].normal   = builder.triNormals[t];
        builder.vertices[t * 3 + k].material = part.triangleMaterials ? part.triangleMaterials[t] : LDR_MATERIALID_INHERIT;
        builder.vertices[t * 3 + k]._pad = 0;
      }
    }

    bool checkMaterials = part.triangleMaterials && config.renderpartVertexMaterials;

    // first is a vertex merge pass over all compatible edges
    for(uint32_t e = 0; e < mesh.numEdges; e++) {

      Mesh::Edge& edge = mesh.edges[e];

      if(edge.isOpen())
        continue;

      uint32_t triLeft  = Mesh::INVALID;
      uint32_t triRight = Mesh::INVALID;
      uint32_t nmList   = Mesh::INVALID;

      while(mesh.iterateTrianglePairs(edge, nmList, triLeft, triRight)) {

        uint32_t leftOrigA = mesh.findTriangleVertex(triLeft, edge.vtxA);
        uint32_t leftOrigB = mesh.findTriangleVertex(triLeft, edge.vtxB);

        uint32_t rightOrigA = mesh.findTriangleVertex(triRight, edge.vtxA);
        uint32_t rightOrigB = mesh.findTriangleVertex(triRight, edge.vtxB);

        mesh.setTriManifoldAdjacency(triLeft, leftOrigA, triRight);
        mesh.setTriManifoldAdjacency(triRight, rightOrigB, triLeft);

        if((edge.flag & EDGE_HARD_BIT))
          continue;

        bool canMergeMaterial = !(checkMaterials) || (part.triangleMaterials[triLeft] == part.triangleMaterials[triLeft]);

        bool canMergeAngle = vec_dot(builder.triNormals[triLeft], builder.triNormals[triRight]) >= Loader::FORCED_HARD_EDGE_DOT;

        if(!canMergeAngle) {
          edge.flag |= EDGE_ANGLE_BIT;
        }

        if(canMergeMaterial && canMergeAngle) {

          uint32_t leftA = triLeft * 3 + leftOrigA;
          uint32_t leftB = triLeft * 3 + leftOrigB;

          uint32_t rightA = triRight * 3 + rightOrigA;
          uint32_t rightB = triRight * 3 + rightOrigB;

          // always use left-side vertex
          builder.vertices[triangleVertices[leftA]].normal =
              vec_add(builder.vertices[triangleVertices[leftA]].normal, builder.vertices[triangleVertices[rightA]].normal);
          builder.vertices[triangleVertices[leftB]].normal =
              vec_add(builder.vertices[triangleVertices[leftB]].normal, builder.vertices[triangleVertices[rightB]].normal);

          // right triangle will use lefts's vertices
          triangleVertices[rightA] = triangleVertices[leftA];
          triangleVertices[rightA] = triangleVertices[leftB];

          // if right is already connected, we need to propagate its connections to point to left
          // triangle vertices now
          for(uint32_t s = 0; s < 2; s++) {
            uint32_t tri      = triRight;
            uint32_t rvtx     = s == 0 ? triangleVertices[leftA] : triangleVertices[leftB];
            uint32_t vtx      = s == 0 ? edge.vtxA : edge.vtxB;
            bool     outgoing = s == 0;
            while(tri != LDR_INVALID_IDX) {
              uint32_t triVtx = mesh.findTriangleVertex(tri, vtx);

              uint32_t triEdge                   = outgoing ? triVtx : (3 + triVtx - 1) % 3;
              triangleVertices[tri * 3 + triVtx] = rvtx;

              Mesh::TriAdjacency triAdjacency = mesh.triAdjacencies[tri];
              uint32_t           newTri       = triAdjacency.edgeSmoothTri[triEdge];
              assert(tri != newTri);
              tri = newTri;
              if(tri == triRight)
                break;
            }
          }

          // connect the triangles
          mesh.setTriSmoothAdjacency(triLeft, leftOrigA, triRight);
          mesh.setTriSmoothAdjacency(triRight, rightOrigB, triLeft);
        }
      }
    }

    // second pass, we must merge normals along edges that were split only due to material
    if(checkMaterials) {
      for(uint32_t e = 0; e < mesh.numEdges; e++) {
        const Mesh::Edge& edge = mesh.edges[e];

        if(edge.isOpen() || (edge.flag & EDGE_HARD_BIT))
          continue;

        uint32_t triLeft  = Mesh::INVALID;
        uint32_t triRight = Mesh::INVALID;
        uint32_t nmList   = Mesh::INVALID;

        while(mesh.iterateTrianglePairs(edge, nmList, triLeft, triRight)) {

          bool canMergeMaterial = part.triangleMaterials[triLeft] == part.triangleMaterials[triLeft];
          bool canMergeAngle = vec_dot(builder.triNormals[triLeft], builder.triNormals[triRight]) >= Loader::FORCED_HARD_EDGE_DOT;

          if(!canMergeMaterial && !canMergeAngle) {

            uint32_t leftOrigA = mesh.findTriangleVertex(triLeft, edge.vtxA);
            uint32_t leftOrigB = mesh.findTriangleVertex(triLeft, edge.vtxB);

            uint32_t leftA = triLeft * 3 + leftOrigA;
            uint32_t leftB = triLeft * 3 + leftOrigB;

            uint32_t rightOrigA = mesh.findTriangleVertex(triRight, edge.vtxA);
            uint32_t rightOrigB = mesh.findTriangleVertex(triRight, edge.vtxB);

            uint32_t rightA = triRight * 3 + rightOrigA;
            uint32_t rightB = triRight * 3 + rightOrigB;

            LdrVector normalA =
                vec_add(builder.vertices[triangleVertices[leftA]].normal, builder.vertices[triangleVertices[rightA]].normal);
            LdrVector normalB =
                vec_add(builder.vertices[triangleVertices[leftB]].normal, builder.vertices[triangleVertices[rightB]].normal);

            builder.vertices[triangleVertices[leftA]].normal  = normalA;
            builder.vertices[triangleVertices[rightA]].normal = normalA;
            builder.vertices[triangleVertices[leftB]].normal  = normalB;
            builder.vertices[triangleVertices[rightB]].normal = normalB;

            mesh.setTriSmoothAdjacency(triLeft, leftOrigA, triRight);
            mesh.setTriSmoothAdjacency(triRight, rightOrigB, triLeft);
          }
        }
      }
    }
    // compact vertices
    Loader::TVector<uint32_t> remapCompact(builder.triangles.size(), LDR_INVALID_IDX);

    uint32_t numVerticesNew = 0;
    for(uint32_t i = 0; i < builder.triangles.size(); i++) {
      if(builder.triangles[i] == i) {
        remapCompact[i] = numVerticesNew++;
      }
    }

    // original vtx in / out

    builder.vtxOutInfo.resize(part.numPositions);

    Loader::TVector<uint32_t> vtxFirstRenderVertex(part.numPositions, LDR_INVALID_IDX);
    Loader::TVector<uint32_t> renderVtxFirstTriangle(numVerticesNew, LDR_INVALID_IDX);

    for(uint32_t i = 0; i < builder.vertices.size(); i++) {
      if(remapCompact[i] != LDR_INVALID_IDX) {
        uint32_t v          = remapCompact[i];
        builder.vertices[v] = builder.vertices[i];

        // also normalize
        builder.vertices[v].normal = vec_normalize(builder.vertices[v].normal);

        uint32_t origVertex = part.triangles[i];
        builder.vtxOutInfo[origVertex].count++;

        vtxFirstRenderVertex[origVertex] = std::min(vtxFirstRenderVertex[origVertex], v);
        renderVtxFirstTriangle[v]        = std::min(renderVtxFirstTriangle[v], i / 3);
      }

      // compact indices
      builder.triangles[i] = remapCompact[builder.triangles[i]];

      assert(builder.triangles[i] < numVerticesNew);
    }


    if(config.renderpartChamfer) {
      builder.vtxOutInfo.resize(part.numPositions);
      builder.vtxOutIndices.resize(numVerticesNew);
      builder.vtxOutEdgePairs.resize(numVerticesNew);

      uint32_t base        = 0;
      uint32_t maxOutCount = 0;
      for(uint32_t i = 0; i < part.numPositions; i++) {
        builder.vtxOutInfo[i].begin = base;
        maxOutCount                 = std::max(maxOutCount, builder.vtxOutInfo[i].count);
        base += builder.vtxOutInfo[i].count;
        builder.vtxOutInfo[i].count = 0;
      }

      for(uint32_t i = 0; i < builder.vertices.size(); i++) {
        if(remapCompact[i] != LDR_INVALID_IDX) {
          uint32_t origVertex = part.triangles[i];
          uint32_t v          = remapCompact[i];
          builder.vtxOutIndices[builder.vtxOutInfo[origVertex].begin + (builder.vtxOutInfo[origVertex].count++)] = v;
        }
      }

      Loader::TVector<uint32_t>                            localOutIdx(maxOutCount, 0);
      Loader::TVector<uint32_t>                            localVertices(maxOutCount, 0);
      Loader::TVector<Loader::BuilderRenderPart::EdgePair> localPairs(maxOutCount);

      for(uint32_t i = 0; i < part.numPositions; i++) {
        Loader::BuilderRenderPart::VertexInfo& outInfo = builder.vtxOutInfo[i];

        if(outInfo.count < 2)
          continue;

        if(i == 182) {
          i = i;
        }

        memset(localOutIdx.data(), LDR_INVALID_IDX, maxOutCount * sizeof(uint32_t));

        for(uint32_t o = 0; o < outInfo.count; o++) {
          uint32_t rvtx = builder.vtxOutIndices[outInfo.begin + o];

          localVertices[o] = rvtx;

          // find the left and right open edges for this vertex if any
          // pure material split may not introduce unconnected edges
          Loader::BuilderRenderPart::EdgePair& pair = localPairs[o];
          pair.pairs[0]                             = LDR_INVALID_IDX;
          pair.pairs[1]                             = LDR_INVALID_IDX;

          uint32_t firstTri = renderVtxFirstTriangle[rvtx];

          uint32_t pairTriEdge[2] = {LDR_INVALID_IDX, LDR_INVALID_IDX};

          // check both side edges
          for(uint32_t s = 0; s < 2; s++) {
            uint32_t tri      = firstTri;
            bool     outgoing = s == 0;
            while(tri != LDR_INVALID_IDX) {
              uint32_t triVtx = mesh.findTriangleVertex(tri, i);

              uint32_t triEdge = outgoing ? triVtx : (3 + triVtx - 1) % 3;
              pair.tri[s]      = tri;
              pairTriEdge[s]   = triEdge;

              Mesh::TriAdjacency triAdjacency = mesh.triAdjacencies[tri];
              uint32_t           newTri       = triAdjacency.edgeSmoothTri[triEdge];
              assert(tri != newTri);
              tri = newTri;
              if(tri == firstTri)
                break;
            }
          }

          for(uint32_t s = 0; s < 2; s++) {
            const uint32_t* vertices = mesh.getTriangle(pair.tri[s]);
            pair.edge[s]             = mesh.getEdgeIdx(vertices[pairTriEdge[s]], vertices[(pairTriEdge[s] + 1) % 3]);
          }

          for(uint32_t p = 0; p < o; p++) {
            Loader::BuilderRenderPart::EdgePair& ppair = localPairs[p];

            // connect pairs with other pairs

            if((ppair.edge[0] == pair.edge[0] || ppair.edge[1] == pair.edge[0])
               && mesh.areTriManifoldAdjacent(pair.tri[0], ppair.edge[0] == pair.edge[0] ? ppair.tri[0] : ppair.tri[1])) {

              pair.pairs[0]                                      = p;
              ppair.pairs[ppair.edge[0] == pair.edge[0] ? 0 : 1] = o;
            }

            if((ppair.edge[0] == pair.edge[1] || ppair.edge[1] == pair.edge[1])
               && mesh.areTriManifoldAdjacent(pair.tri[1], ppair.edge[0] == pair.edge[1] ? ppair.tri[0] : ppair.tri[1])) {

              pair.pairs[1]                                      = p;
              ppair.pairs[ppair.edge[0] == pair.edge[1] ? 0 : 1] = o;
            }
          }
        }

        // build connected groups

        uint32_t newOutCount = 0;

        for(uint32_t s = 0; s < 2; s++) {
          // first pass is find those that are not dual connected.
          // otherwise can start at any

          for(uint32_t o = 0; o < outInfo.count; o++) {
            if(localOutIdx[o] == LDR_INVALID_IDX
               && ((s == 0 && localPairs[o].pairs[0] == LDR_INVALID_IDX || localPairs[o].pairs[1] == LDR_INVALID_IDX) || s)) {
              // walk onesided
              localOutIdx[o]      = newOutCount++;
              localPairs[o].group = outInfo.groupCount;

              uint32_t prev = o;
              uint32_t next = localPairs[o].pairs[0] == LDR_INVALID_IDX ? localPairs[o].pairs[1] : localPairs[o].pairs[0];
              while(next != LDR_INVALID_IDX) {
                //assert(localOutIdx[next] == LDR_INVALID_IDX);
                if(localOutIdx[next] == LDR_INVALID_IDX) {
                  localPairs[next].group = outInfo.groupCount;
                  localOutIdx[next]      = newOutCount++;
                }
                else {
                  // TODO fixme, should not get here
                  break;
                }
                uint32_t tmp = next;
                next = localPairs[next].pairs[0] == prev ? localPairs[next].pairs[1] : localPairs[next].pairs[0];
                prev = tmp;
                // circle
                if(next == o)
                  break;
              }

              outInfo.groupCount++;
            }
          }
        }

        for(uint32_t o = 0; o < outInfo.count; o++) {
          if(localOutIdx[o] == LDR_INVALID_IDX) {
            localPairs[o].group = outInfo.groupCount++;
            localOutIdx[o]      = newOutCount++;
          }
        }

        // output re-ordered
        for(uint32_t o = 0; o < outInfo.count; o++) {
          uint32_t oIdx                                 = localOutIdx[o];
          builder.vtxOutIndices[outInfo.begin + oIdx]   = localVertices[o];
          builder.vtxOutEdgePairs[outInfo.begin + oIdx] = localPairs[o];
        }
      }
    }

    builder.vertices.resize(numVerticesNew);

    for(uint32_t i = 0; i < part.numLines; i++) {
      uint32_t vertexA = part.lines[i * 2 + 0];
      uint32_t vertexB = part.lines[i * 2 + 1];

      uint32_t renderVertexA = vtxFirstRenderVertex[vertexA];

      if(renderVertexA == LDR_INVALID_IDX) {
        renderVertexA                 = builder.vertices.size();
        vtxFirstRenderVertex[vertexA] = renderVertexA;
        builder.vertices.push_back(make_vertex(part.positions[vertexA]));
      }

      uint32_t renderVertexB = vtxFirstRenderVertex[vertexB];
      if(renderVertexB == LDR_INVALID_IDX) {
        renderVertexB                 = builder.vertices.size();
        vtxFirstRenderVertex[vertexB] = renderVertexB;
        builder.vertices.push_back(make_vertex(part.positions[vertexB]));
      }

      builder.lines.push_back(renderVertexA);
      builder.lines.push_back(renderVertexB);
    }
  }

  static void buildRenderPart(Loader::BuilderRenderPart& builder, const LdrPart& part, const Loader::Config& config)
  {
    builder.bbox = part.bbox;
    builder.flag = part.flag;

    Mesh mesh;
    mesh.initFull(part.numPositions, part.numTriangles, part.triangles, part.positions, builder.loader);

    builder.vertices.reserve(part.numPositions + part.numLines * 2);
    builder.lines.reserve(part.numLines * 2);
    builder.triangles.resize(part.numTriangles * 3, LDR_INVALID_IDX);

    builder.triNormals.reserve(part.numTriangles);

    for(uint32_t t = 0; t < part.numTriangles; t++) {
      builder.triNormals.push_back(mesh.getTriangleNormal(t, part.positions));
    }

    // push "hard" lines
    for(uint32_t i = 0; i < part.numLines; i++) {
      Mesh::Edge* edge = mesh.getEdge(part.lines[i * 2 + 0], part.lines[i * 2 + 1]);
      if(edge) {
        edge->flag |= EDGE_HARD_BIT;
      }
    }

    buildRenderPartBasic(builder, mesh, part, config);

    if(config.renderpartChamfer) {
      chamferRenderPart(mesh, builder, part, config.renderpartChamfer);
    }
  }
};


//////////////////////////////////////////////////////////////////////////

static LdrMaterial getDefaultMaterial()
{
  LdrMaterial material  = {0};
  material.baseColor[0] = 0xFF;
  material.baseColor[1] = 0xFF;
  material.baseColor[2] = 0xFF;
  material.alpha        = 0xFF;
  material.type         = LDR_MATERIAL_DEFAULT;
  strncpy(material.name, "default", sizeof(material.name));
  return material;
}

Loader::Loader(const LdrLoaderCreateInfo* config)
{
  LdrResult result = init(config);
  assert(result == LDR_SUCCESS);
}

LdrResult Loader::init(const LdrLoaderCreateInfo* createInfo)
{
  LdrLoaderCreateInfo* pCfg = (LdrLoaderCreateInfo*)&m_config;

  *pCfg                   = *createInfo;
  m_config.basePathString = createInfo->basePath;
  if(createInfo->cacheFile) {
    m_config.cachFileString = createInfo->cacheFile;
  }
  m_config.basePath  = nullptr;
  m_config.cacheFile = nullptr;

  m_searchPaths[0] = m_config.basePathString + "/p/48/";
  m_searchPaths[1] = m_config.basePathString + "/p/";
  static_assert(2 == PRIMITIVE_PATHS, "primitive paths not correct");
  m_searchPaths[2] = m_config.basePathString + "/parts/";
  m_searchPaths[3] = m_config.basePathString + "/unofficial/";

  // FIXME should count ldraw files
  // we reserve to get fixed memory pointers for parts
  m_parts.reserve(MAX_PARTS);
  m_primitives.reserve(MAX_PRIMS);
  m_renderParts.resize(MAX_PARTS, {});

  m_partRegistry.reserve(MAX_PARTS + MAX_PRIMS);

  m_startedPartLoad.resize(MAX_PARTS, false);
  m_startedPrimitiveLoad.resize(MAX_PRIMS, false);
  m_startedPartRenderBuild.resize(MAX_PARTS, false);

#if LDR_CFG_THREADSAFE
  m_finishedPartLoad.resize(MAX_PARTS, false);
  m_finishedPrimitiveLoad.resize(MAX_PRIMS, false);
  m_finishedPartRenderBuild.resize(MAX_PARTS, false);
#endif

  // load material config
  Text txt;

  LdrMaterialType type = LDR_MATERIAL_SOLID;

  if(txt.load((m_config.basePathString + "/LDConfig.ldr").c_str())) {
    char* line = txt.buffer;
    for(size_t i = 0; i < txt.size; i++) {
      if(txt.buffer[i] == '\r')
        txt.buffer[i] = '\n';
      if(txt.buffer[i] != '\n')
        continue;
      txt.buffer[i] = 0;

      if(strstr(line, "0 // LDraw Solid Colours")) {
        type = LDR_MATERIAL_SOLID;
      }
      else if(strstr(line, "0 // LDraw Transparent Colours")) {
        type = LDR_MATERIAL_TRANSPARENT;
      }
      else if(strstr(line, "0 // LDraw Chrome Colours")) {
        type = LDR_MATERIAL_CHROME;
      }
      else if(strstr(line, "0 // LDraw Pearl Colours")) {
        type = LDR_MATERIAL_PEARL;
      }
      else if(strstr(line, "0 // LDraw Metallic Colours")) {
        type = LDR_MATERIAL_METALLIC;
      }
      else if(strstr(line, "0 // LDraw Milky Colours")) {
        type = LDR_MATERIAL_MILKY;
      }
      else if(strstr(line, "0 // LDraw Glitter Colours")) {
        type = LDR_MATERIAL_GLITTER;
      }
      else if(strstr(line, "0 // LDraw Speckle Colours")) {
        type = LDR_MATERIAL_SPECKLE;
      }
      else if(strstr(line, "0 // LDraw Rubber Colours")) {
        type = LDR_MATERIAL_RUBBER;
      }
      else if(strstr(line, "0 // LDraw Internal Common Material Colours")) {
        type = LDR_MATERIAL_COMMON;
      }

      LdrMaterial material = {};
      material.type        = type;
      uint32_t code        = 0;
      uint32_t value       = 0;
      uint32_t edge        = 0;
      if(4 == sscanf(line, "0 !COLOUR %47s CODE %d VALUE #%X EDGE #%X", material.name, &code, &value, &edge)) {
        material.baseColor[2] = (value >> 0) & 0xFF;
        material.baseColor[1] = (value >> 8) & 0xFF;
        material.baseColor[0] = (value >> 16) & 0xFF;
        material.edgeColor[2] = (edge >> 0) & 0xFF;
        material.edgeColor[1] = (edge >> 8) & 0xFF;
        material.edgeColor[0] = (edge >> 16) & 0xFF;
        uint32_t temp         = 0;

        code = fixupMaterialID(code);

        const char* search = nullptr;
        search             = strstr(line, "ALPHA");
        if(search && 1 == sscanf(search, "ALPHA %d", &temp)) {
          material.alpha = temp;
        }
        search = strstr(line, "LUMINANCE");
        if(search && 1 == sscanf(search, "LUMINANCE %d", &temp)) {
          material.emissive = temp;
        }
        search = strstr(line, "GLITTER");
        if(search
           && 4
                  == sscanf(search, "GLITTER VALUE #%X FRACTION %f VFRACTION %f SIZE %f", &temp,
                            &material.glitter.fraction, &material.glitter.vfraction, &material.glitter.size)) {
          material.glitter.color[2] = (temp >> 0) & 0xFF;
          material.glitter.color[1] = (temp >> 8) & 0xFF;
          material.glitter.color[0] = (temp >> 16) & 0xFF;
          material.type             = LDR_MATERIAL_GLITTER;
        }
        search = strstr(line, "SPECKLE");
        if(search
           && 4
                  == sscanf(search, "SPECKLE VALUE #%X FRACTION %f MINSIZE %f MAXSIZE %f", &temp,
                            &material.speckle.fraction, &material.speckle.minsize, &material.speckle.maxsize)) {
          material.speckle.color[2] = (temp >> 0) & 0xFF;
          material.speckle.color[1] = (temp >> 8) & 0xFF;
          material.speckle.color[0] = (temp >> 16) & 0xFF;
          material.type             = LDR_MATERIAL_SPECKLE;
        }

        if(strstr(line, "METAL")) {
          material.type = LDR_MATERIAL_METALLIC;
        }
        if(strstr(line, "RUBBER")) {
          material.type = LDR_MATERIAL_RUBBER;
        }
        if(strstr(line, "CHROME")) {
          material.type = LDR_MATERIAL_CHROME;
        }
        if(strstr(line, "PEARLESCENT")) {
          material.type = LDR_MATERIAL_PEARL;
        }
      }

      if(m_materials.size() <= code) {
        m_materials.resize(code + 1, getDefaultMaterial());
      }

      m_materials[code] = material;


      txt.buffer[i] = '\n';
      line          = &txt.buffer[i + 1];
    }
  }

  return LDR_SUCCESS;
}

void Loader::deinit()
{
  for(size_t i = 0; i < m_parts.size(); i++) {
    deinitPart(m_parts[i]);
  }
  for(size_t i = 0; i < m_primitives.size(); i++) {
    deinitPart(m_primitives[i]);
  }
  // use part sizes for renderparts
  for(size_t i = 0; i < m_parts.size(); i++) {
    deinitRenderPart(m_renderParts[i]);
  }

  m_primitives.clear();
  m_parts.clear();
  m_renderParts.clear();

  m_startedPartLoad.clear();
  m_startedPrimitiveLoad.clear();
  m_startedPartRenderBuild.clear();

#if LDR_CFG_THREADSAFE
  m_finishedPartLoad.clear();
  m_finishedPrimitiveLoad.clear();
  m_finishedPartRenderBuild.clear();
#endif

  m_partRegistry.clear();
  m_shapeRegistry.clear();
  m_materials.clear();
}

LdrResult Loader::registerInternalPart(const char* filename, const std::string& foundname, bool isPrimitive, bool startLoad, PartEntry& entry)
{
  LdrResult result = LDR_SUCCESS;

#if LDR_CFG_THREADSAFE
  std::lock_guard<std::mutex> lockguard(m_partRegistryMutex);
#endif
  PartEntry newentry;
  if(isPrimitive) {
    newentry.primID = m_primitives.size();
    if(newentry.primID == MAX_PRIMS) {
      return LDR_ERROR_RESERVED_MEMORY;
    }
  }
  else {
    newentry.partID = m_parts.size();
    if(newentry.partID == MAX_PARTS) {
      return LDR_ERROR_RESERVED_MEMORY;
    }
  }

  const auto it = m_partRegistry.insert({filename, newentry});
  if(it.second) {
    entry  = newentry;
    result = LDR_SUCCESS;
    if(isPrimitive) {
      m_primitives.push_back({LDR_ERROR_INITIALIZATION});
      m_primitiveFoundnames.push_back(foundname);
      if(startLoad) {
        startPrimitive(entry.primID);
      }
    }
    else {
      m_parts.push_back({LDR_ERROR_INITIALIZATION});
      m_partFoundnames.push_back(foundname);
      if(startLoad) {
        startPart(entry.partID);
      }
      // load will take care of those, so block these actions
      if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
        startBuildRender(entry.partID);
      }
    }
  }
  else {
    entry  = it.first->second;
    result = LDR_WARNING_IN_USE;
  }

  return result;
}

bool Loader::findLibraryFile(const char* filename, std::string& foundname, bool allowPrimitives, bool& isPrimitive)
{
  for(uint32_t i = allowPrimitives ? (m_config.partHiResPrimitives ? 0 : 1) : PRIMITIVE_PATHS; i < SEARCH_PATHS; i++) {
    foundname = m_searchPaths[i] + filename;

    FILE* f     = fopen(foundname.c_str(), "rb");
    bool  found = f != nullptr;
    if(f) {
      fclose(f);
      isPrimitive = i < PRIMITIVE_PATHS;
      if(allowPrimitives && SUBPART_AS_PRIMITIVE && filename[0] == 's' && (filename[1] == '/' || filename[1] == '\\'))
        isPrimitive = true;
      return true;
    }
  }

  return false;
}

LdrResult Loader::deferPart(const char* filename, bool allowPrimitives, PartEntry& entry)
{
  bool found = findEntry(filename, entry) == LDR_SUCCESS;
  if(!found) {
    // try register
    std::string foundname;
    bool        isPrimitive;
    if(findLibraryFile(filename, foundname, allowPrimitives, isPrimitive)) {
      LdrResult result = registerInternalPart(filename, foundname, isPrimitive, false, entry);
      if(result == LDR_WARNING_IN_USE) {
        // someone else was faster at registration
        if(entry.isPrimitive() != isPrimitive) {
          return LDR_ERROR_INVALID_OPERATION;
        }
        else {
          return LDR_SUCCESS;
        }
      }
      else {
        return result;
      }
    }
    else {
      return LDR_ERROR_FILE_NOT_FOUND;
    }
  }

  return LDR_SUCCESS;
}

LdrResult Loader::resolvePart(const char* filename, bool allowPrimitives, PartEntry& entry, uint32_t depth)
{
  bool found = findEntry(filename, entry) == LDR_SUCCESS;

  if(!found) {
    // try register
    std::string foundname;
    bool        isPrimitive = false;
    if(findLibraryFile(filename, foundname, allowPrimitives, isPrimitive)) {
      LdrResult result = registerInternalPart(filename, foundname, isPrimitive, true, entry);
      if(result == LDR_SUCCESS) {
        // we are the first, let's load the part
        if(isPrimitive) {
          result = loadData(m_primitives[entry.primID], m_renderParts[entry.primID], foundname.c_str(), isPrimitive, depth);
          signalPrimitive(entry.primID);
        }
        else {
          result = loadData(m_parts[entry.partID], m_renderParts[entry.partID], foundname.c_str(), isPrimitive, depth);
          signalPart(entry.partID);
          if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
            signalBuildRender(entry.partID);
          }
        }
        return result;
      }
      else if(result == LDR_WARNING_IN_USE) {
        // someone else was faster at registration
        found = true;
        if(entry.isPrimitive() != isPrimitive) {
          return LDR_ERROR_INVALID_OPERATION;
        }
      }
      else {
        return result;
      }
    }
    else {
#ifdef _DEBUG
      fprintf(stderr, "file not found: %s\n", filename);
#endif
      return LDR_ERROR_FILE_NOT_FOUND;
    }
  }

  if(found) {
    // wait until loading completed / start loading
    if(entry.isPrimitive()) {
      waitPrimitive(entry.primID);
      return m_primitives[entry.primID].loadResult;
    }
    else {
      // deferred loads could mean we have not started loading despite having found something
      if(startPartAction(entry.partID)) {
        loadData(m_parts[entry.partID], m_renderParts[entry.partID], m_partFoundnames[entry.partID].c_str(), false, depth);
        signalPart(entry.partID);
        if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
          signalBuildRender(entry.partID);
        }
      }

      waitPart(entry.partID);
      return m_parts[entry.partID].loadResult;
    }
  }
  return LDR_ERROR_INVALID_OPERATION;
}

LdrResult Loader::resolveRenderPart(LdrPartID partid)
{
  waitPart(partid);

  bool built = false;
  if(startBuildRenderAction(partid)) {
    buildRenderPart(partid);
  }
  else {
    waitBuildRender(partid);
  }

  return LDR_SUCCESS;
}


LdrResult Loader::registerShapeType(const char* filename, LdrShapeType type)
{
  if(type == LDR_INVALID_SHAPETYPE) {
    return LDR_ERROR_INVALID_OPERATION;
  }

  bool inserted = m_shapeRegistry.insert({filename, type}).second;

  return inserted ? LDR_SUCCESS : LDR_ERROR_INVALID_OPERATION;
}

LdrResult Loader::registerPrimitive(const char* filename, const LdrPart* part)
{
  PartEntry entry;
  LdrResult result = registerInternalPart(filename, filename, true, true, entry);

  if(result == LDR_SUCCESS) {
    m_primitives[entry.primID] = *part;
    signalPrimitive(entry.primID);
    return LDR_SUCCESS;
  }

  return result == LDR_WARNING_IN_USE ? LDR_ERROR_INVALID_OPERATION : result;
}

LdrResult Loader::registerPart(const char* filename, const LdrPart* part, LdrPartID* pPartID)
{
  PartEntry entry;
  LdrResult result = registerInternalPart(filename, filename, false, true, entry);

  if(result == LDR_SUCCESS) {
    m_parts[entry.partID] = *part;
    signalPart(entry.partID);

    *pPartID = entry.partID;
    return LDR_SUCCESS;
  }

  *pPartID = LDR_INVALID_ID;
  return result == LDR_WARNING_IN_USE ? LDR_ERROR_INVALID_OPERATION : result;
}

LdrResult Loader::registerRenderPart(LdrPartID partid, const LdrRenderPart* rpart)
{
  if(partid >= getNumRegisteredParts()) {
    return LDR_ERROR_INVALID_OPERATION;
  }

  m_renderParts[partid] = *rpart;
  signalBuildRender(partid);

  return LDR_SUCCESS;
}

LdrResult Loader::rawAllocate(size_t size, LdrRawData* raw)
{
  raw->size = size;
  raw->data = size ? _mm_malloc(size, 16) : nullptr;
  return LDR_SUCCESS;
}

LdrResult Loader::rawFree(const LdrRawData* raw)
{
  if(raw->data) {
    _mm_free(raw->data);
  }
  return LDR_SUCCESS;
}

LdrResult Loader::preloadPart(const char* filename, LdrPartID* pPartID)
{
  PartEntry entry;
  LdrResult res = resolvePart(filename, false, entry, 0);
  if(res == LDR_SUCCESS) {
    *pPartID = entry.partID;
  }
  else {
    *pPartID = LDR_INVALID_ID;
  }

  return res;
}

LdrResult Loader::buildRenderParts(uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  if(m_config.partFixMode != LDR_RENDERPART_BUILD_ONDEMAND)
    return LDR_ERROR_INVALID_OPERATION;

  bool inFlight = false;
  bool all      = false;

  if(parts == nullptr) {
    numParts = getNumRegisteredParts();
    all      = true;
  }

  const uint8_t* partsBytes = (const uint8_t*)parts;
  for(uint32_t i = 0; i < numParts; i++) {
    LdrPartID partid = all ? (LdrPartID)i : *(const LdrPartID*)(partsBytes + i * partStride);
    if(startBuildRenderAction(partid)) {
      buildRenderPart(partid);
    }
    else {
      inFlight = true;
    }
  }

  if(inFlight) {
    for(uint32_t i = 0; i < numParts; i++) {
      LdrPartID partid = all ? (LdrPartID)i : *(const LdrPartID*)(partsBytes + i * partStride);
      waitBuildRender(partid);
    }
  }

  return LDR_SUCCESS;
}


LdrResult Loader::loadDeferredParts(uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  LdrResult resultFinal = LDR_SUCCESS;
  bool      inFlight    = false;
  bool      all         = false;

  if(numParts == ~0 && parts == nullptr) {
    numParts = getNumRegisteredParts();
    all      = true;
  }

  for(uint32_t i = 0; i < numParts; i++) {
    LdrPartID partID = all ? (LdrPartID)i : *(const LdrPartID*)(((const uint8_t*)parts) + partStride * i);
    if(startPartAction(partID)) {
      LdrResult result = loadData(m_parts[partID], m_renderParts[partID], m_partFoundnames[partID].c_str(), false, 0);
      signalPart(partID);
      if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
        signalBuildRender(partID);
      }
      if(result != LDR_SUCCESS) {
        resultFinal = result;
      }
    }
    else {
      inFlight = true;
    }
  }

  if(inFlight) {
    for(uint32_t i = 0; i < numParts; i++) {
      LdrPartID partID = all ? (LdrPartID)i : *(const LdrPartID*)(((const uint8_t*)parts) + partStride * i);
      waitPart(partID);

      LdrResult result = m_parts[partID].loadResult;
      if(result != LDR_SUCCESS) {
        resultFinal = result;
      }
    }
  }

  if(resultFinal == LDR_ERROR_FILE_NOT_FOUND) {
    resultFinal = LDR_WARNING_PART_NOT_FOUND;
  }

  return resultFinal;
}

LdrResult Loader::createModel(const char* filename, LdrBool32 autoResolve, LdrModelHDL* pModel)
{
  LdrModel* model  = new LdrModel;
  LdrResult result = loadModel(*model, filename, autoResolve);
  if(result == LDR_SUCCESS || result == LDR_WARNING_PART_NOT_FOUND) {
    *pModel = model;
  }
  else {
    delete model;
  }
  return result;
}

LdrResult Loader::resolveModel(LdrModelHDL model)
{
  BuilderModel builder;
  builder.bbox = model->bbox;

  LdrResult result = LDR_SUCCESS;

  for(uint32_t i = 0; i < model->numInstances; i++) {
    const LdrInstance& instance = model->instances[i];
    const LdrPart&     part     = getPart(instance.part);
    if(part.loadResult != LDR_SUCCESS) {
      result = part.loadResult;
    }

    if(part.numShapes || part.numPositions) {
      builder.instances.push_back(instance);
      bbox_merge(builder.bbox, instance.transform, getPart(instance.part).bbox);
    }
    // flatten
    for(uint32_t s = 0; s < part.numInstances; s++) {
      LdrInstance subinstance = part.instances[s];
      if(subinstance.material == LDR_MATERIALID_INHERIT) {
        subinstance.material = instance.material;
      }
      subinstance.transform = mat_mul(instance.transform, subinstance.transform);
      builder.instances.push_back(subinstance);
      bbox_merge(builder.bbox, subinstance.transform, getPart(subinstance.part).bbox);
    }
  }

  deinitModel(*(LdrModel*)model);
  initModel(*(LdrModel*)model, builder);

  return result;
}

void Loader::destroyModel(LdrModelHDL model)
{
  if(model) {
    deinitModel(*(LdrModel*)model);
    delete(LdrModel*)model;
  }
}

LdrResult Loader::createRenderModel(LdrModelHDL model, LdrBool32 autoResolve, LdrRenderModelHDL* pRenderModel)
{
  LdrRenderModel* renderModel = new LdrRenderModel;
  LdrResult       result      = makeRenderModel(*renderModel, model, autoResolve);
  if(result == LDR_SUCCESS || result == LDR_WARNING_PART_NOT_FOUND) {
    *pRenderModel = renderModel;
  }
  else {
    delete renderModel;
  }

  *pRenderModel = renderModel;
  return LDR_SUCCESS;
}

void Loader::destroyRenderModel(LdrRenderModelHDL renderModel)
{
  if(renderModel) {
    deinitRenderModel(*(LdrRenderModel*)renderModel);
    delete(LdrRenderModel*)renderModel;
  }
}

LdrResult Loader::loadData(LdrPart& part, LdrRenderPart& renderPart, const char* filename, bool isPrimitive, uint32_t depth)
{
  Text txt;
  if(!txt.load(filename)) {
    // actually should not get here, as resolve function takes care of finding files
    assert(0);

    part.loadResult = LDR_ERROR_FILE_NOT_FOUND;
    return part.loadResult;
  }
#if defined(_DEBUG) && (LDR_DEBUG_FLAG & LDR_DEBUG_FLAG_FILELOAD)
  const char* depth_prefix = "...................";
  uint32_t    clampDepth   = (uint32_t)strlen(depth_prefix);
  printf("%sldr load data %s\n", &depth_prefix[clampDepth - std::min(depth, clampDepth)], filename);
#endif

  BuilderPart builder;
  builder.flag.isPrimitive = isPrimitive ? 1 : 0;
  builder.filename         = filename;
#ifdef _DEBUG
  builder.loader = this;
#endif

  BFCWinding   winding        = BFC_CCW;
  BFCCertified certified      = BFC_UNKNOWN;
  bool         localCull      = true;
  bool         invertNext     = false;
  bool         keepInvertNext = false;

  std::string subfilenameLong;
  char        subfilenameShort[512] = {0};

  char* line = txt.buffer;
  for(size_t i = 0; i < txt.size; i++) {
    if(txt.buffer[i] == '\r')
      txt.buffer[i] = '\n';
    if(txt.buffer[i] != '\n')
      continue;

    txt.buffer[i] = 0;

    size_t lineLength = &txt.buffer[i] - line;

    // parse line

    int dummy;
    int material = LDR_MATERIALID_INHERIT;

    int typ = atoi(line);

    switch(typ) {
      case 0: {
        // we only care for BFC
        char* bfc = strstr(line, "0 BFC");
        if(bfc) {
          if(certified == BFC_UNKNOWN) {
            certified = BFC_TRUE;
          }
          if(strstr(bfc, "CERTIFY")) {
            certified = BFC_TRUE;
          }
          if(strstr(bfc, "NOCERTIFY")) {
            certified = BFC_FALSE;
          }
          if(strstr(bfc, "CLIP")) {
            localCull = true;
          }
          if(strstr(bfc, "NOCLIP")) {
            localCull = false;
          }
          if(strstr(bfc, "CW")) {
            winding = BFC_CW;
          }
          if(strstr(bfc, "CCW")) {
            winding = BFC_CCW;
          }
          if(strstr(bfc, "INVERTNEXT")) {
            invertNext     = true;
            keepInvertNext = true;
          }
        }
        if(strstr(line, "0 !LDRAW_ORG Subpart")) {
          builder.flag.isSubpart = 1;
        }
      } break;
      case 1: {
        // sub file
        LdrMatrix transform;
        float*    mat = transform.values;
        mat[3] = mat[7] = mat[11] = 0;
        mat[15]                   = 1.0f;

        char* subfilename = subfilenameShort;
        if(lineLength >= sizeof(subfilenameShort)) {
          subfilenameLong = std::string(lineLength, 0);
          subfilename     = &subfilenameLong[0];
        }

        int read = sscanf(line, "%d %i %f %f %f %f %f %f %f %f %f %f %f %f %[^\n\t]", &dummy, &material, mat + 12, mat + 13,
                          mat + 14, mat + 0, mat + 4, mat + 8, mat + 1, mat + 5, mat + 9, mat + 2, mat + 6, mat + 10, subfilename);

        if(read != 15) {
#ifdef _DEBUG
          fprintf(stderr, "ldr parser error: %s - %s\n", filename, line);
#endif
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        material = fixupMaterialID(material);

        // find existing part/primitive
        LdrShapeType shapeType = m_shapeRegistry.empty() ? LDR_INVALID_ID : findShapeType(subfilename);
        if(shapeType != LDR_INVALID_ID) {
          // add to shape vector
          LdrShape shape;
          shape.transform = transform;
          shape.material  = material;
          shape.type      = shapeType;
          shape.bfcInvert = invertNext ? 1 : 0;
          builder.shapes.push_back(shape);
        }
        else {
          PartEntry entry;
          LdrResult result = resolvePart(subfilename, true, entry, depth + 1);
          if(result == LDR_SUCCESS) {
            if(entry.isPrimitive()) {
              appendBuilderPrimitive(builder, transform, entry.primID, material, invertNext);
            }
            else if(m_parts[entry.partID].flag.isSubpart) {
              appendBuilderSubPart(builder, transform, entry.partID, material, invertNext);
            }
            else {
              appendBuilderPart(builder, transform, entry.partID, material, invertNext);
            }
          }
          else {
            part.loadResult = result;
            return part.loadResult;
          }
        }

        keepInvertNext = false;
        // append to builder
      } break;
      case 2: {
        LdrVector vecA;
        LdrVector vecB;
        // line
        int read =
            sscanf(line, "%d %i %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z, &vecB.x, &vecB.y, &vecB.z);
        if(read != 8) {
#ifdef _DEBUG
          fprintf(stderr, "ldr parser error: %s - %s\n", filename, line);
#endif
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        material = fixupMaterialID(material);

        uint32_t vidx = (uint32_t)builder.positions.size();
        builder.lines.push_back(vidx);
        builder.lines.push_back(vidx + 1);
        builder.positions.push_back(vecA);
        builder.positions.push_back(vecB);
        //assert(material == LDR_MATERIALID_EDGE);

        builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
      } break;
      case 3: {
        // triangle
        LdrVector vecA;
        LdrVector vecB;
        LdrVector vecC;
        // line
        int read = sscanf(line, "%d %i %f %f %f %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z,
                          &vecB.x, &vecB.y, &vecB.z, &vecC.x, &vecC.y, &vecC.z);

        if(read != 11) {
#ifdef _DEBUG
          fprintf(stderr, "ldr parser error: %s - %s\n", filename, line);
#endif
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        material = fixupMaterialID(material);

        float distA = vec_length(vec_sub(vecB, vecA));
        float distB = vec_length(vec_sub(vecC, vecB));
        float distC = vec_length(vec_sub(vecA, vecC));
        float dist  = std::min(std::min(distA, distB), distC);

        float dotA = fabsf(vec_dot(vec_normalize(vec_sub(vecB, vecA)), vec_normalize(vec_sub(vecC, vecA))));
        if(dotA <= Loader::NO_AREA_TRIANGLE_DOT && dist > Loader::MIN_MERGE_EPSILON) {
          uint32_t vidx = (uint32_t)builder.positions.size();
          uint32_t tidx = (uint32_t)builder.triangles.size() / 3;
          // normalize to ccw
          if(winding == BFC_CW) {
            builder.triangles.push_back(vidx + 2);
            builder.triangles.push_back(vidx + 1);
            builder.triangles.push_back(vidx + 0);
          }
          else {
            builder.triangles.push_back(vidx);
            builder.triangles.push_back(vidx + 1);
            builder.triangles.push_back(vidx + 2);
          }

          builder.positions.push_back(vecA);
          builder.positions.push_back(vecB);
          builder.positions.push_back(vecC);
          builder.triangleMaterials.push_back(material);

          LdrNgon ngon;
          ngon.num   = 1;
          ngon.index = tidx;
          builder.triangleNgons.push_back(ngon);

          //builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
          //builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecB, vecC)));
          //builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecC, vecA)));
        }
      } break;
      case 4: {
        // quad
        LdrVector vecA;
        LdrVector vecB;
        LdrVector vecC;
        LdrVector vecD;
        // line
        int read = sscanf(line, "%d %i %f %f %f %f %f %f %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y,
                          &vecA.z, &vecB.x, &vecB.y, &vecB.z, &vecC.x, &vecC.y, &vecC.z, &vecD.x, &vecD.y, &vecD.z);
        if(read != 14) {
#ifdef _DEBUG
          fprintf(stderr, "ldr parser error: %s - %s\n", filename, line);
#endif
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        material = fixupMaterialID(material);

        float dotA = fabsf(vec_dot(vec_normalize(vec_sub(vecB, vecA)), vec_normalize(vec_sub(vecC, vecA))));
        float dotD = fabsf(vec_dot(vec_normalize(vec_sub(vecA, vecD)), vec_normalize(vec_sub(vecC, vecD))));
        if(dotA <= Loader::NO_AREA_TRIANGLE_DOT || dotD <= Loader::NO_AREA_TRIANGLE_DOT) {
          uint32_t vidx = (uint32_t)builder.positions.size();
          uint32_t tidx = (uint32_t)builder.triangles.size() / 3;

          // flip edge based on angles

          uint32_t quad[6] = {0, 1, 2, 2, 3, 0};

          if(Loader::ALLOW_QUAD_EDGEFLIP && dotD < dotA) {
            quad[2] = 3;
            quad[5] = 1;
          }

          // normalize to ccw
          if(winding == BFC_CW) {
            builder.triangles.push_back(vidx + quad[5]);
            builder.triangles.push_back(vidx + quad[4]);
            builder.triangles.push_back(vidx + quad[3]);
            builder.triangles.push_back(vidx + quad[2]);
            builder.triangles.push_back(vidx + quad[1]);
            builder.triangles.push_back(vidx + quad[0]);
          }
          else {
            builder.triangles.push_back(vidx + quad[0]);
            builder.triangles.push_back(vidx + quad[1]);
            builder.triangles.push_back(vidx + quad[2]);
            builder.triangles.push_back(vidx + quad[3]);
            builder.triangles.push_back(vidx + quad[4]);
            builder.triangles.push_back(vidx + quad[5]);
          }
          builder.positions.push_back(vecA);
          builder.positions.push_back(vecB);
          builder.positions.push_back(vecC);
          builder.positions.push_back(vecD);
          builder.triangleMaterials.push_back(material);
          builder.triangleMaterials.push_back(material);

          LdrNgon ngon;
          ngon.num   = 2;
          ngon.index = tidx;
          builder.triangleNgons.push_back(ngon);
          builder.triangleNgons.push_back(ngon);
        }
      } break;
      case 5: {
        // optional line
        LdrVector vecA;
        LdrVector vecB;
        // line
        int read =
            sscanf(line, "%d %i %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z, &vecB.x, &vecB.y, &vecB.z);
        if(read != 8) {
#ifdef _DEBUG
          fprintf(stderr, "ldr parser error: %s - %s\n", filename, line);
#endif
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        material = fixupMaterialID(material);

        uint32_t vidx = (uint32_t)builder.positions.size();
        builder.optional_lines.push_back(vidx);
        builder.optional_lines.push_back(vidx + 1);
        builder.positions.push_back(vecA);
        builder.positions.push_back(vecB);

        builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
      } break;
    }

    if((typ == 1 || typ == 3 || typ == 4) && material != LDR_MATERIALID_INHERIT) {
      builder.flag.hasComplexMaterial = 1;
    }
    if((typ == 3 || typ == 4) && (!localCull || certified != BFC_TRUE)) {
      // inconsistent clipping state for this part
      builder.flag.hasNoBackFaceCulling = 1;
    }

    if(!keepInvertNext) {
      invertNext = false;
    }

    txt.buffer[i] = '\n';
    line          = txt.buffer + (i + 1);
  }

  for(size_t i = 0; i < builder.positions.size(); i++) {
    bbox_merge(builder.bbox, builder.positions[i]);
  }
  compactBuilderPart(builder);

  if(!isPrimitive && m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
    MeshUtils::fixBuilderPart(builder, m_config);
  }

  initPart(part, builder);
  part.loadResult = LDR_SUCCESS;

  if((!isPrimitive && !builder.flag.isSubpart) && m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
    LdrPart partTemp = part;

    if(!m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
      MeshUtils::fixBuilderPart(builder, m_config);
      initPart(partTemp, builder);
    }

    BuilderRenderPart builderRender;
    builderRender.loader   = this;
    builderRender.filename = part.name;
    MeshUtils::buildRenderPart(builderRender, partTemp, m_config);
    initRenderPart(renderPart, builderRender, partTemp);

    if(!m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
      deinitPart(partTemp);
    }
  }

  return part.loadResult;
}


LdrResult Loader::appendSubModel(BuilderModel& builder, Text& txt, const LdrMatrix& transform, LdrMaterialID material, LdrBool32 autoResolve, uint32_t depth)
{
  LdrResult finalResult = LDR_SUCCESS;

  std::string subfilenameLong;
  char        subfilenameShort[512] = {0};

  char* line = txt.buffer;
  for(size_t i = 0; i < txt.size; i++) {
    if(txt.buffer[i] == '\r')
      txt.buffer[i] = '\n';
    if(txt.buffer[i] != '\n')
      continue;
    txt.buffer[i] = 0;

    size_t lineLength = &txt.buffer[i] - line;

    // parse line
    int typ = atoi(line);

    // only interested in parts
    if(typ == 1) {
      LdrInstance instance;
      float*      mat = instance.transform.values;
      mat[3] = mat[7] = mat[11] = 0;
      mat[15]                   = 1.0f;

      char* subfilename = subfilenameShort;
      if(lineLength >= sizeof(subfilenameShort)) {
        subfilenameLong = std::string(lineLength, 0);
        subfilename     = &subfilenameLong[0];
      }

      int materialRead;

      int read = sscanf(line, "%d %i %f %f %f %f %f %f %f %f %f %f %f %f %[^\n\t]", &typ, &materialRead, mat + 12, mat + 13,
                        mat + 14, mat + 0, mat + 4, mat + 8, mat + 1, mat + 5, mat + 9, mat + 2, mat + 6, mat + 10, subfilename);

      instance.material = fixupMaterialID(materialRead);

      instance.transform = mat_mul(transform, instance.transform);
      if(instance.material == LDR_MATERIALID_INHERIT) {
        instance.material = material;
      }

      if(read == 15) {
        // look in subfiles
        bool found = false;
        for(size_t i = 0; i < builder.subFilenames.size(); i++) {
          if(builder.subFilenames[i] == std::string(subfilename)) {
            LdrResult result =
                appendSubModel(builder, builder.subTexts[i], instance.transform, instance.material, autoResolve, depth + 1);
            if(result == LDR_SUCCESS) {
              found = true;
            }
            else if(result == LDR_WARNING_PART_NOT_FOUND || result == LDR_ERROR_FILE_NOT_FOUND) {
              found       = true;
              finalResult = LDR_WARNING_PART_NOT_FOUND;
            }
            else {
              return result;
            }
          }
        }
        // look in library
        if(!found) {
          PartEntry entry;
          LdrResult result =
              autoResolve ? resolvePart(subfilename, false, entry, depth + 1) : deferPart(subfilename, false, entry);
          if(result == LDR_SUCCESS) {
            instance.part = entry.partID;
            assert(instance.part != LDR_INVALID_ID);
            builder.instances.push_back(instance);

            if(autoResolve) {
              const LdrPart& part = getPart(instance.part);

              if(part.numShapes || part.numPositions) {
                bbox_merge(builder.bbox, instance.transform, getPart(instance.part).bbox);
              }
              // flatten
              for(uint32_t s = 0; s < part.numInstances; s++) {
                LdrInstance subinstance = part.instances[s];
                if(subinstance.material == LDR_MATERIALID_INHERIT) {
                  subinstance.material = instance.material;
                }
                subinstance.transform = mat_mul(instance.transform, subinstance.transform);
                assert(subinstance.part != LDR_INVALID_ID);
                builder.instances.push_back(subinstance);
                bbox_merge(builder.bbox, subinstance.transform, getPart(subinstance.part).bbox);
              }
            }
          }
          else if(result == LDR_WARNING_PART_NOT_FOUND || result == LDR_ERROR_FILE_NOT_FOUND) {
            finalResult = LDR_WARNING_PART_NOT_FOUND;
          }
          else {
            return result;
          }
        }
      }
      else {
        return LDR_ERROR_PARSER;
      }
    }
    txt.buffer[i] = '\n';
    line          = &txt.buffer[i + 1];
  }

  return finalResult;
}

LdrResult Loader::loadModel(LdrModel& model, const char* filename, LdrBool32 autoResolve)
{
  Text txt;
  if(!txt.load(filename)) {
    return LDR_ERROR_FILE_NOT_FOUND;
  }

  std::string subfilenameLong;
  char        subfilenameShort[512] = {0};

  BuilderModel builder;
  // split into sub-texts
  bool isMpd = false;
  if(strstr(filename, ".mpd") || strstr(filename, ".MPD")) {
    isMpd = true;

    std::string nextFilename;

    char* nextFileBegin = txt.buffer;
    char* line          = txt.buffer;

    for(size_t i = 0; i < txt.size; i++) {
      if(txt.buffer[i] == '\r')
        txt.buffer[i] = '\n';
      if(txt.buffer[i] != '\n')
        continue;
      txt.buffer[i] = 0;

      size_t lineLength = &txt.buffer[i] - line;

      // parse line
      int typ = atoi(line);

      if(typ == 0) {

        char* subfilename = subfilenameShort;
        if(lineLength >= sizeof(subfilenameShort)) {
          subfilenameLong = std::string(lineLength, 0);
          subfilename     = &subfilenameLong[0];
        }

        int read = sscanf(line, "0 FILE %[^\n\t]", subfilename);
        if(read == 1) {
          if(nextFileBegin && nextFileBegin != line) {
            Text subTxt;
            subTxt.buffer     = nextFileBegin;
            subTxt.size       = line - nextFileBegin;
            subTxt.referenced = true;

            builder.subFilenames.push_back(nextFilename);
            builder.subTexts.push_back(subTxt);
          }
          nextFilename  = subfilename;
          nextFileBegin = line;
        }
      }

      txt.buffer[i] = '\n';
      line          = &txt.buffer[i + 1];
    }

    // terminate
    {
      Text subTxt;
      subTxt.buffer     = nextFileBegin;
      subTxt.size       = (txt.buffer + txt.size) - nextFileBegin;
      subTxt.referenced = true;

      builder.subFilenames.push_back(nextFilename);
      builder.subTexts.push_back(subTxt);
    }
  }

  LdrMatrix transform = mat_identity();
  LdrResult finalResult =
      appendSubModel(builder, isMpd ? builder.subTexts[0] : txt, transform, LDR_MATERIALID_INHERIT, autoResolve, 0);

  initModel(model, builder);
  return finalResult;
}

LdrResult Loader::makeRenderModel(LdrRenderModel& rmodel, LdrModelHDL model, LdrBool32 autoResolve)
{
  BuilderRenderModel builder;

  builder.bbox = model->bbox;
  for(uint32_t i = 0; i < model->numInstances; i++) {
    const LdrInstance&    instance = model->instances[i];
    BuilderRenderInstance rinstance;
    rinstance.instance = instance;

    if(autoResolve) {
      // try to build on-demand
      LdrResult result = resolveRenderPart(instance.part);
      if(result != LDR_SUCCESS) {
        return result;
      }
    }

    waitBuildRender(instance.part);
    builder.instances.push_back(rinstance);
  }

  initRenderModel(rmodel, builder);

  return LDR_SUCCESS;
}


class Utils
{
public:
  static inline void pushWithOffset(Loader::TVector<LdrVertexIndex>& indices, uint32_t offset, uint32_t num, const LdrVertexIndex* src)
  {
    for(uint32_t i = 0; i < num; i++) {
      indices.push_back(src[i] + offset);
    }
  }

  static inline void pushWithOffsetRev3(Loader::TVector<LdrVertexIndex>& indices, uint32_t offset, uint32_t num, const LdrVertexIndex* src)
  {
    for(uint32_t i = 0; i < num; i++) {
      indices.push_back(src[i * 3 + 2] + offset);
      indices.push_back(src[i * 3 + 1] + offset);
      indices.push_back(src[i * 3 + 0] + offset);
    }
  }

  static inline void pushWithOffsetRev4(Loader::TVector<LdrVertexIndex>& indices, uint32_t offset, uint32_t num, const LdrVertexIndex* src)
  {
    for(uint32_t i = 0; i < num; i++) {
      indices.push_back(src[i * 4 + 3] + offset);
      indices.push_back(src[i * 4 + 2] + offset);
      indices.push_back(src[i * 4 + 1] + offset);
      indices.push_back(src[i * 4 + 0] + offset);
    }
  }

  template <class T>
  static inline void copyAppend(Loader::TVector<T>& dst, uint32_t num, const T* src)
  {
    if(num) {
      size_t begin = dst.size();
      dst.resize(dst.size() + num);
      memcpy(dst.data() + begin, src, sizeof(T) * num);
    }
  }

  static size_t applyRemapTriangles(size_t                       numIndices,
                                    LdrVertexIndex*              indices,
                                    LdrMaterialID*               triangleMaterials,
                                    LdrNgon*                     triangleNgons,
                                    const uint32_t* LDR_RESTRICT remapVertices)
  {
    size_t   outIndices   = 0;
    size_t   outTri       = 0;
    size_t   numTri       = numIndices / 3;
    uint32_t skipped      = 0;
    bool     prevWasValid = false;
    for(size_t t = 0; t < numTri; t++) {
      LdrNgon       origNgon     = triangleNgons[t];
      LdrMaterialID origMaterial = triangleMaterials[t];
      uint32_t      orig[3];
      orig[0] = indices[t * 3 + 0];
      orig[1] = indices[t * 3 + 1];
      orig[2] = indices[t * 3 + 2];

      uint32_t newIdx[3];
      newIdx[0] = remapVertices[orig[0]];
      newIdx[1] = remapVertices[orig[1]];
      newIdx[2] = remapVertices[orig[2]];

      // degenerated triangle
      if(newIdx[0] == newIdx[1] || newIdx[1] == newIdx[2] || newIdx[2] == newIdx[0]) {
        assert(origNgon.num == 0 || origNgon.num == 2);

        // tell other triangle it's no longer a quad
        if(origNgon.num == 2) {
          // we are the first, then disable state for next
          if(origNgon.index == t) {
            triangleNgons[t + 1] = {};
          }  // we are second, then disable for previous
          else if(origNgon.index == t - 1 && prevWasValid) {
            triangleNgons[outTri - 1] = {};
          }
        }

        skipped++;
        prevWasValid = false;
        continue;
      }

      indices[outIndices + 0]   = newIdx[0];
      indices[outIndices + 1]   = newIdx[1];
      indices[outIndices + 2]   = newIdx[2];
      triangleMaterials[outTri] = origMaterial;
      if(origNgon.num) {
        origNgon.index -= skipped;
      }
      triangleNgons[outTri] = origNgon;

      outIndices += 3;
      outTri++;
      prevWasValid = true;
    }

    return outIndices;
  }

  static size_t applyRemapLines(size_t numIndices, LdrVertexIndex* indices, const uint32_t* LDR_RESTRICT remapVertices)
  {
    size_t outIndices = 0;
    for(size_t i = 0; i < numIndices / 2; i++) {
      uint32_t orig[2];
      orig[0] = indices[i * 2 + 0];
      orig[1] = indices[i * 2 + 1];

      uint32_t newIdx[2];
      newIdx[0] = remapVertices[orig[0]];
      newIdx[1] = remapVertices[orig[1]];

      if(newIdx[0] == newIdx[1])
        continue;

      indices[outIndices + 0] = newIdx[0];
      indices[outIndices + 1] = newIdx[1];

      outIndices += 2;
    }

    return outIndices;
  }

  template <class T>
  static void fillVector(Loader::TVector<T>& vec, const T* data, uint32_t num)
  {
    if(data) {
      vec.resize(num);
      memcpy(vec.data(), data, sizeof(T) * num);
    }
  }

  inline static void alignSize(LdrRawData& raw, size_t alignSz)
  {
    size_t rest = raw.size % alignSz;
    if(rest) {
      raw.size += alignSz - rest;
    }
  }

  template <class Tout>
  static uint32_t append(LdrRawData& raw, Tout*& ptrRef, size_t num)
  {
    assert(num <= 0xFFFFFFFF);
    alignSize(raw, alignof(Tout));
    ptrRef = (Tout*)raw.size;
    raw.size += sizeof(Tout) * num;
    return uint32_t(num);
  }

  template <class T, class Tout>
  static uint32_t append(LdrRawData& raw, Tout*& ptrRef, const Loader::TVector<T>& vec, size_t divisor = 1)
  {
    assert((vec.size() / divisor) <= 0xFFFFFFFF);
    alignSize(raw, alignof(Tout));
    ptrRef = (Tout*)raw.size;
    raw.size += sizeof(Tout) * vec.size();
    return uint32_t(vec.size() / divisor);
  }

  template <class T, class Tout>
  static void setup_pointer(LdrRawData& raw, Tout*& ptrRef, const Loader::TVector<T>& vec)
  {
    if(vec.empty()) {
      ptrRef = nullptr;
    }
    else {
      size_t offset = (size_t)ptrRef;
      ptrRef        = (Tout*)((uint8_t*)raw.data + offset);
      memcpy(ptrRef, vec.data(), vec.size() * sizeof(Tout));
    }

    static_assert(sizeof(Tout) == sizeof(T), "builder type mismatches size");
  }

  template <class T, class Tout>
  static void setup_pointer(LdrRawData& raw, Tout*& ptrRef, size_t num, const T* data)
  {
    if(num == 0) {
      ptrRef = nullptr;
    }
    else {
      size_t offset = (size_t)ptrRef;
      ptrRef        = (Tout*)((uint8_t*)raw.data + offset);
      memcpy(ptrRef, data, num * sizeof(Tout));
    }

    static_assert(sizeof(Tout) == sizeof(T), "builder type mismatches size");
  }

  template <class Tout>
  static void setup_pointer(LdrRawData& raw, Tout*& ptrRef, size_t offset, bool valid = true)
  {
    if(valid) {
      ptrRef = (Tout*)((uint8_t*)raw.data + offset);
    }
    else {
      ptrRef = nullptr;
    }
  }

  template <class Tout>
  static void setup_pointer(LdrRawData& raw, Tout*& ptrRef, bool valid = true)
  {
    if(valid) {
      size_t offset = (size_t)ptrRef;
      ptrRef        = (Tout*)((uint8_t*)raw.data + offset);
    }
    else {
      ptrRef = nullptr;
    }
  }
};


void Loader::appendBuilderPart(BuilderPart& builder, const LdrMatrix& transform, LdrPartID partid, LdrMaterialID material, bool flipWinding)
{
  assert(flipWinding == false);
  const LdrPart& part = getPart(partid);
  if(part.numPositions || part.numShapes) {
    LdrInstance instance;
    instance.material  = material;
    instance.part      = partid;
    instance.transform = transform;
    builder.instances.push_back(instance);
  }
  // flatten sub parts
  for(uint32_t s = 0; s < part.numInstances; s++) {
    LdrInstance subinstance = part.instances[s];
    subinstance.transform   = mat_mul(transform, subinstance.transform);
    if(subinstance.material == LDR_MATERIALID_INHERIT) {
      subinstance.material = material;
    }
    builder.instances.push_back(subinstance);
  }
}

void Loader::appendBuilderPrimitive(BuilderPart& builder, const LdrMatrix& transform, LdrPrimitiveID primid, LdrMaterialID material, bool flipWinding)
{
  appendBuilderEmbed(builder, transform, m_primitives[primid], material, flipWinding);
}

void Loader::appendBuilderSubPart(BuilderPart& builder, const LdrMatrix& transform, LdrPartID partid, LdrMaterialID material, bool flipWinding)
{
  appendBuilderEmbed(builder, transform, m_parts[partid], material, flipWinding);
}

void Loader::appendBuilderEmbed(BuilderPart& builder, const LdrMatrix& transform, const LdrPart& part, LdrMaterialID material, bool flipWinding)
{
  builder.flag.hasComplexMaterial |= part.flag.hasComplexMaterial;
  builder.flag.hasNoBackFaceCulling |= part.flag.hasNoBackFaceCulling;

  float scale = std::min(std::min(vec_length(make_vec(transform.col[0])), vec_length(make_vec(transform.col[1]))),
                         vec_length(make_vec(transform.col[2])));

  builder.minEdgeLength = std::min(builder.minEdgeLength, part.minEdgeLength * scale);

  bool transformFlip = mat_determinant(transform) < 0;
  flipWinding        = flipWinding ^ transformFlip;

  uint32_t vecOffset   = (uint32_t)builder.positions.size();
  size_t   shapeOffset = (uint32_t)builder.shapes.size();

  builder.positions.reserve(builder.positions.size() + part.numPositions);

  builder.lines.reserve(builder.lines.size() + part.numLines * 2);
  builder.triangles.reserve(builder.triangles.size() + part.numTriangles * 3);
  builder.optional_lines.reserve(builder.optional_lines.size() + part.numOptionalLines * 2);
  builder.triangleNgons.reserve(builder.triangleNgons.size() + part.numTriangles);

  for(uint32_t i = 0; i < part.numPositions; i++) {
    LdrVector vec = transform_point(transform, part.positions[i]);
    builder.positions.push_back(vec);
    bbox_merge(builder.bbox, vec);
  }

  uint32_t triOffset = builder.triangles.size() / 3;
  for(uint32_t i = 0; i < part.numTriangles; i++) {
    LdrNgon ngon = part.triangleNgons[i];
    ngon.index += triOffset;
    builder.triangleNgons.push_back(ngon);
  }

  if(flipWinding) {
    Utils::pushWithOffsetRev3(builder.triangles, vecOffset, part.numTriangles, part.triangles);
  }
  else {
    Utils::pushWithOffset(builder.triangles, vecOffset, part.numTriangles * 3, part.triangles);
  }
  Utils::pushWithOffset(builder.lines, vecOffset, part.numLines * 2, part.lines);
  Utils::pushWithOffset(builder.optional_lines, vecOffset, part.numOptionalLines * 2, part.optional_lines);

  for(uint32_t i = 0; i < part.numTriangles; i++) {
    builder.triangleMaterials.push_back(part.triangleMaterials[i] == LDR_MATERIALID_INHERIT ? material : part.triangleMaterials[i]);
  }

  // shapes
  Utils::copyAppend(builder.shapes, part.numShapes, part.shapes);
  for(size_t i = shapeOffset; i < builder.shapes.size(); i++) {
    builder.shapes[i].bfcInvert ^= flipWinding ? 1 : 0;
    builder.shapes[i].transform = mat_mul(transform, builder.shapes[i].transform);
  }

  // flatten sub parts
  for(uint32_t s = 0; s < part.numInstances; s++) {
    LdrInstance subinstance = part.instances[s];
    subinstance.transform   = mat_mul(transform, subinstance.transform);
    if(subinstance.material == LDR_MATERIALID_INHERIT) {
      subinstance.material = material;
    }
    builder.instances.push_back(subinstance);
  }
}

struct SortVertex
{
  float     dot;
  LdrVector pos;
  uint32_t  idx;
};

static bool SortVertex_cmp(const SortVertex& a, const SortVertex& b)
{
  return a.dot < b.dot;
}

inline size_t runMerge(SortVertex& refVertex, SortVertex* LDR_RESTRICT vertices, size_t begin, size_t end, uint32_t* LDR_RESTRICT remap, float epsilon)
{
  size_t newBegin = 0;
  for(size_t i = begin; i <= end; i++) {
    SortVertex& vertex = vertices[i];
    if(vertex.idx != LDR_INVALID_ID) {
      bool canMerge = vec_sq_length(vec_sub(vertex.pos, refVertex.pos)) <= epsilon * epsilon;
      if(canMerge) {
        remap[vertex.idx] = refVertex.idx;
        vertex.idx        = LDR_INVALID_ID;
      }
      else if(!newBegin) {
        newBegin = i + 1;
      }
    }
  }
  if(!newBegin) {
    newBegin = end + 1;
  }
  refVertex = vertices[newBegin - 1];

  return newBegin;
}

void Loader::compactBuilderPart(BuilderPart& builder)
{
  if(builder.positions.empty())
    return;

  TVector<uint32_t> remapMerge(builder.positions.size());
  TVector<uint32_t> remapCompact(builder.positions.size());

  // algorithm inspired by http://www.assimp.org/
  // sort points along single direction, then line sweep merge

  LdrVector normal = vec_normalize(vec_sub(builder.bbox.max, builder.bbox.min));

  size_t numVertices = builder.positions.size();

  std::vector<SortVertex> sortedVertices;
  sortedVertices.reserve(numVertices);
  for(size_t i = 0; i < numVertices; i++) {
    SortVertex svertex;
    svertex.pos = builder.positions[i];
    svertex.idx = (uint32_t)i;
    svertex.dot = vec_dot(normal, svertex.pos);
    sortedVertices.push_back(svertex);
    remapMerge[i] = i;
  }

  std::sort(sortedVertices.begin(), sortedVertices.end(), SortVertex_cmp);

  // merge using minimal edge length
  const float epsilon = MIN_MERGE_EPSILON;  //  std::min(MIN_MERGE_EPSILON, builder.minEdgeLength * 0.9f);

  // line sweep merge
  size_t     mergeBegin = 1;
  SortVertex refVertex  = sortedVertices[0];

  for(size_t i = 1; i < numVertices; i++) {
    const SortVertex& svertex = sortedVertices[i];
    while(mergeBegin <= i && ((svertex.dot - refVertex.dot > epsilon) || (i == numVertices - 1))) {
      mergeBegin = runMerge(refVertex, sortedVertices.data(), mergeBegin, i, remapMerge.data(), epsilon);
    }
  }

  uint32_t numVerticesNew = 0;
  for(size_t i = 0; i < numVertices; i++) {
    if(remapMerge[i] == i) {
      remapCompact[i] = numVerticesNew++;
    }
    else {
      remapCompact[i] = LDR_INVALID_ID;
    }
  }

  for(size_t i = 0; i < numVertices; i++) {
    if(remapCompact[i] != LDR_INVALID_ID) {
      builder.positions[remapCompact[i]] = builder.positions[i];
    }
    remapMerge[i] = remapCompact[remapMerge[i]];

    assert(remapMerge[i] < numVerticesNew);
    assert(remapMerge[i] != LDR_INVALID_ID);
  }

  builder.positions.resize(numVerticesNew);

#if _DEBUG
#if 0
    float minDist = FLT_MAX;

    for (size_t i = 0; i < numVerticesNew; i++)
    {
      for (size_t v = i + 1; v < numVerticesNew; v++)
      {
        float sqlen = vec_sq_length(vec_sub(builder.positions[i], builder.positions[v]));
        if (sqlen <= epsilon * epsilon) {
          i = i;
        }
        minDist = std::min(minDist, sqlen);
      }
    }
    minDist = sqrtf(minDist);
#endif
#endif

  size_t lineSize = Utils::applyRemapLines(builder.lines.size(), builder.lines.data(), remapMerge.data());
  size_t optionalSize = Utils::applyRemapLines(builder.optional_lines.size(), builder.optional_lines.data(), remapMerge.data());
  size_t triangleSize =
      Utils::applyRemapTriangles(builder.triangles.size(), builder.triangles.data(), builder.triangleMaterials.data(),
                                 builder.triangleNgons.data(), remapMerge.data());

  builder.lines.resize(lineSize);
  builder.optional_lines.resize(optionalSize);
  builder.triangles.resize(triangleSize);
  builder.triangleMaterials.resize(triangleSize / 3);
  builder.triangleNgons.resize(triangleSize / 3);
}


void Loader::fillBuilderPart(BuilderPart& builder, LdrPartID partid)
{
  LdrPart& part = m_parts[partid];

  builder.filename      = std::string(part.name);
  builder.flag          = part.flag;
  builder.bbox          = part.bbox;
  builder.minEdgeLength = part.minEdgeLength;
  Utils::fillVector(builder.positions, part.positions, part.numPositions);
  Utils::fillVector(builder.lines, part.lines, part.numLines * 2);
  Utils::fillVector(builder.optional_lines, part.optional_lines, part.numOptionalLines * 2);
  Utils::fillVector(builder.triangles, part.triangles, part.numTriangles * 3);
  Utils::fillVector(builder.triangleMaterials, part.triangleMaterials, part.numTriangles);
  Utils::fillVector(builder.triangleNgons, part.triangleNgons, part.numTriangles);
  Utils::fillVector(builder.instances, part.instances, part.numInstances);
  Utils::fillVector(builder.shapes, part.shapes, part.numShapes);
}

void Loader::fixPart(LdrPartID partid)
{
  LdrPart& part = m_parts[partid];

  if(!(part.flag.isPrimitive || part.flag.isSubpart)) {
    BuilderPart builder;
#ifdef _DEBUG
    builder.loader = this;
#endif
    fillBuilderPart(builder, partid);

    deinitPart(part);
    memset(&part, 0, sizeof(LdrPart));

    MeshUtils::fixBuilderPart(builder, m_config);

    initPart(part, builder);
  }
}

void Loader::buildRenderPart(LdrPartID partid)
{
  LdrRenderPart& renderPart = m_renderParts[partid];

  if(m_parts[partid].flag.isPrimitive || m_parts[partid].flag.isSubpart) {
    renderPart = LdrRenderPart();
  }
  else {
    BuilderRenderPart builder;
    builder.loader   = this;
    builder.filename = m_parts[partid].name;
    if(m_config.partFixMode == LDR_PART_FIX_NONE) {
      LdrPart parttemp;

      BuilderPart builderPart;
      builderPart.loader = this;
      fillBuilderPart(builderPart, partid);
      MeshUtils::fixBuilderPart(builderPart, m_config);
      initPart(parttemp, builderPart);

      MeshUtils::buildRenderPart(builder, parttemp, m_config);
      initRenderPart(renderPart, builder, parttemp);
      deinitPart(parttemp);
    }
    else {
      MeshUtils::buildRenderPart(builder, m_parts[partid], m_config);
      initRenderPart(renderPart, builder, m_parts[partid]);
    }
  }

  signalBuildRender(partid);
}


#define EXTRA_ASSERTS 0

void Loader::initPart(LdrPart& part, const BuilderPart& builder)
{
#if _DEBUG && EXTRA_ASSERTS
  for(size_t i = 0; i < builder.lines.size(); i++) {
    assert(builder.lines[i] <= builder.positions.size());
  }
  for(size_t i = 0; i < builder.optional_lines.size(); i++) {
    assert(builder.optional_lines[i] <= builder.positions.size());
  }
  for(size_t i = 0; i < builder.triangles.size(); i++) {
    assert(builder.triangles[i] <= builder.positions.size());
  }
  assert(builder.triangles.size() == builder.materials.size() * 3);
#endif


  part.flag          = builder.flag;
  part.bbox          = builder.bbox;
  part.minEdgeLength = builder.minEdgeLength;

  part.raw.size = 0;

  part.numPositions     = Utils::append(part.raw, part.positions, builder.positions);
  part.numLines         = Utils::append(part.raw, part.lines, builder.lines, 2);
  part.numOptionalLines = Utils::append(part.raw, part.optional_lines, builder.optional_lines, 2);
  part.numTriangles     = Utils::append(part.raw, part.triangles, builder.triangles, 3);
  Utils::append(part.raw, part.triangleNgons, builder.triangleNgons);
  Utils::append(part.raw, part.triangleMaterials, builder.triangleMaterials);
  part.numShapes    = Utils::append(part.raw, part.shapes, builder.shapes);
  part.numInstances = Utils::append(part.raw, part.instances, builder.instances);
  part.numName      = Utils::append(part.raw, part.name, builder.filename.size() + 1);

  rawAllocate(part.raw.size, &part.raw);

  Utils::setup_pointer(part.raw, part.positions, builder.positions);
  Utils::setup_pointer(part.raw, part.lines, builder.lines);
  Utils::setup_pointer(part.raw, part.optional_lines, builder.optional_lines);
  Utils::setup_pointer(part.raw, part.triangles, builder.triangles);
  Utils::setup_pointer(part.raw, part.triangleNgons, builder.triangleNgons);
  Utils::setup_pointer(part.raw, part.triangleMaterials, builder.triangleMaterials);
  Utils::setup_pointer(part.raw, part.shapes, builder.shapes);
  Utils::setup_pointer(part.raw, part.instances, builder.instances);
  Utils::setup_pointer(part.raw, part.name, builder.filename.size() + 1, builder.filename.c_str());

#if _DEBUG && EXTRA_ASSERTS
  assert(builder.positions.size() == part.numPositions);
  assert(builder.lines.size() == part.numLines * 2);
  for(size_t i = 0; i < builder.lines.size(); i++) {
    assert(part.lines[i] <= builder.positions.size());
  }
  assert(builder.optional_lines.size() == part.numOptionalLines * 2);
  for(size_t i = 0; i < builder.optional_lines.size(); i++) {
    assert(part.optional_lines[i] <= builder.positions.size());
  }
  assert(builder.triangles.size() == part.numTriangles * 3);
  for(size_t i = 0; i < builder.triangles.size(); i++) {
    assert(part.triangles[i] <= builder.positions.size());
  }
#endif
}

void Loader::initModel(LdrModel& model, const BuilderModel& builder)
{
  model.bbox         = builder.bbox;
  model.raw.size     = 0;
  model.numInstances = Utils::append(model.raw, model.instances, builder.instances);

  rawAllocate(model.raw.size, &model.raw);

  Utils::setup_pointer(model.raw, model.instances, builder.instances);
}

void Loader::initRenderPart(LdrRenderPart& renderpart, const BuilderRenderPart& builder, const LdrPart& part)
{
  uint32_t keepMaterials = part.flag.hasComplexMaterial ? 1 : 0;  //  && m_config.renderpartTriangleMaterials
  renderpart.flag        = builder.flag;
  renderpart.bbox        = builder.bbox;
  renderpart.raw.size    = 0;

  renderpart.numVertices   = Utils::append(renderpart.raw, renderpart.vertices, builder.vertices);
  renderpart.numLines      = Utils::append(renderpart.raw, renderpart.lines, builder.lines, 2);
  renderpart.numTriangles  = Utils::append(renderpart.raw, renderpart.triangles, builder.triangles, 3);
  renderpart.numTrianglesC = Utils::append(renderpart.raw, renderpart.trianglesC, builder.trianglesC, 3);
  Utils::append(renderpart.raw, renderpart.triangleMaterials, renderpart.numTriangles * keepMaterials);
  Utils::append(renderpart.raw, renderpart.materialsC, renderpart.numTrianglesC * keepMaterials);
  renderpart.numShapes = Utils::append(renderpart.raw, renderpart.shapes, part.numShapes);

  rawAllocate(renderpart.raw.size, &renderpart.raw);

  Utils::setup_pointer(renderpart.raw, renderpart.vertices, builder.vertices);
  Utils::setup_pointer(renderpart.raw, renderpart.lines, builder.lines);
  Utils::setup_pointer(renderpart.raw, renderpart.triangles, builder.triangles);
  Utils::setup_pointer(renderpart.raw, renderpart.trianglesC, builder.trianglesC);
  Utils::setup_pointer(renderpart.raw, renderpart.triangleMaterials, part.numTriangles * keepMaterials, part.triangleMaterials);
  Utils::setup_pointer(renderpart.raw, renderpart.materialsC, builder.materialsC);
  Utils::setup_pointer(renderpart.raw, renderpart.shapes, part.numShapes, part.shapes);
}

void Loader::initRenderModel(LdrRenderModel& rmodel, const BuilderRenderModel& builder)
{
  rmodel.raw.size     = 0;
  rmodel.numInstances = Utils::append(rmodel.raw, rmodel.instances, builder.instances.size());
  rmodel.bbox         = builder.bbox;

  size_t begin = rmodel.raw.size;

  rawAllocate(rmodel.raw.size, &rmodel.raw);

  Utils::setup_pointer(rmodel.raw, rmodel.instances, 0, true);

  for(uint32_t i = 0; i < rmodel.numInstances; i++) {
    const LdrRenderPart& part      = getRenderPart(builder.instances[i].instance.part);
    LdrRenderInstance&   rinstance = rmodel.instances[i];
    rinstance.instance             = builder.instances[i].instance;
  }
}

void Loader::deinitPart(LdrPart& part)
{
  rawFree(&part.raw);
  part.raw = {0, 0};
}

void Loader::deinitRenderPart(LdrRenderPart& part)
{
  rawFree(&part.raw);
  part.raw = {0, 0};
}

void Loader::deinitModel(LdrModel& model)
{
  rawFree(&model.raw);
  model.raw = {0, 0};
}

void Loader::deinitRenderModel(LdrRenderModel& model)
{
  rawFree(&model.raw);
  model.raw = {0, 0};
}

bool Loader::BuilderPart::isSameTriangle(uint32_t tA, uint32_t tB) const
{
  const uint32_t* trisA = &triangles[tA * 3];
  const uint32_t* trisB = &triangles[tB * 3];

  for(uint32_t v = 0; v < 3; v++) {
    if(trisA[0] == trisB[v]) {
      for(uint32_t i = 1; i < 3; i++) {
        if(trisA[i] != trisB[(v + i) % 3])
          return false;
      }
      return true;
    }
  }

  return false;
}

bool Loader::BuilderPart::isSameQuad(uint32_t tA, uint32_t tB) const
{
  if(!isQuad(tA) || !isQuad(tB)) {
    return false;
  }

  uint32_t quadA[4];
  uint32_t quadB[4];

  getCanonicalQuad(tA, quadA);
  getCanonicalQuad(tB, quadB);

  for(uint32_t v = 0; v < 4; v++) {
    if(quadA[0] == quadB[v]) {
      for(uint32_t i = 1; i < 4; i++) {
        if(quadA[i] != quadB[(v + i) % 4])
          return false;
      }
      return true;
    }
  }

  return false;
}

void Loader::BuilderPart::getCanonicalQuad(uint32_t t, uint32_t quad[4]) const
{
  assert(isQuad(t));

  uint32_t first  = triangleNgons[t].index;
  uint32_t second = first + 1;
  quad[0]         = triangles[first * 3 + 0];
  quad[1]         = triangles[first * 3 + 1];
  quad[2]         = triangles[second * 3 + 0];
  quad[3]         = triangles[second * 3 + 1];
}

}  // namespace ldr

//////////////////////////////////////////////////////////////////////////
// C-Api

#if LDR_CFG_C_API

LDR_API void ldrGetDefaultCreateInfo(LdrLoaderCreateInfo* info)
{
  memset(info, 0, sizeof(LdrLoaderCreateInfo));
  info->partHiResPrimitives = LDR_FALSE;
  info->partFixMode         = LDR_PART_FIX_NONE;
  info->partFixTjunctions   = LDR_TRUE;
  info->partFixOverlap      = LDR_FALSE;
  info->renderpartBuildMode = LDR_RENDERPART_BUILD_ONLOAD;
  info->renderpartChamfer   = 0.35f;
  //info->renderpartTriangleMaterials = LDR_TRUE;
  info->renderpartVertexMaterials = LDR_TRUE;
}

LDR_API LdrResult ldrCreateLoader(const LdrLoaderCreateInfo* info, LdrLoaderHDL* pLoader)
{
  ldr::Loader* loader = new ldr::Loader;
  LdrResult    result = loader->init(info);

  if(result == LDR_SUCCESS) {
    *pLoader = (LdrLoaderHDL)loader;
  }
  else {
    delete loader;
  }

  return result;
}
LDR_API void ldrDestroyLoader(LdrLoaderHDL loader)
{
  if(loader) {
    ldr::Loader* lldr = (ldr::Loader*)loader;
    lldr->deinit();
    delete lldr;
  }
}

LDR_API LdrResult ldrRegisterShapeType(LdrLoaderHDL loader, const char* filename, LdrShapeType type)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerShapeType(filename, type);
}

LDR_API LdrResult ldrRegisterPart(LdrLoaderHDL loader, const char* filename, const LdrPart* part, LdrPartID* pPartID)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerPart(filename, part, pPartID);
}

LDR_API LdrResult ldrRegisterPrimitive(LdrLoaderHDL loader, const char* filename, const LdrPart* part)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerPrimitive(filename, part);
}

LDR_API LdrResult ldrRegisterRenderPart(LdrLoaderHDL loader, LdrPartID partId, const LdrRenderPart* rpart)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerRenderPart(partId, rpart);
}

LDR_API LdrResult ldrRawAllocate(LdrLoaderHDL loader, size_t size, LdrRawData* raw)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->rawAllocate(size, raw);
}

LDR_API LdrResult ldrRawFree(LdrLoaderHDL loader, const LdrRawData* raw)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->rawFree(raw);
}

LDR_API LdrResult ldrPreloadPart(LdrLoaderHDL loader, const char* filename, LdrPartID* pPartID)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->preloadPart(filename, pPartID);
}

LDR_API LdrResult ldrBuildRenderParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->buildRenderParts(numParts, parts, partStride);
}

LDR_API LdrResult ldrCreateModel(LdrLoaderHDL loader, const char* filename, LdrBool32 autoResolve, LdrModelHDL* pModel)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->createModel(filename, autoResolve, pModel);
}

LDR_API void ldrResolveModel(LdrLoaderHDL loader, LdrModelHDL model)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  lldr->resolveModel(model);
}

LDR_API void ldrDestroyModel(LdrLoaderHDL loader, LdrModelHDL model)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  lldr->destroyModel(model);
}

LDR_API LdrResult ldrCreateRenderModel(LdrLoaderHDL loader, LdrModelHDL model, LdrBool32 autoResolve, LdrRenderModelHDL* pRenderModel)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->createRenderModel(model, autoResolve, pRenderModel);
}
LDR_API void ldrDestroyRenderModel(LdrLoaderHDL loader, LdrRenderModelHDL renderModel)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  lldr->destroyRenderModel(renderModel);
}

LDR_API LdrResult ldrLoadDeferredParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->loadDeferredParts(numParts, parts, partStride);
}

LDR_API LdrPartID ldrFindPart(LdrLoaderHDL loader, const char* filename)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->findPart(filename);
}

LDR_API LdrPrimitiveID ldrFindPrimitive(LdrLoaderHDL loader, const char* filename)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->findPrimitive(filename);
}

LDR_API uint32_t ldrGetNumRegisteredParts(LdrLoaderHDL loader)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getNumRegisteredParts();
}

LDR_API uint32_t ldrGetNumRegisteredMaterials(LdrLoaderHDL loader)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getNumRegisteredMaterials();
}

LDR_API const LdrMaterial* ldrGetMaterial(LdrLoaderHDL loader, LdrMaterialID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getMaterialP(idx);
}
LDR_API const LdrPart* ldrGetPart(LdrLoaderHDL loader, LdrPartID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getPartP(idx);
}
LDR_API const LdrPart* ldrGetPrimitive(LdrLoaderHDL loader, LdrPrimitiveID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getPrimitiveP(idx);
}
LDR_API const LdrRenderPart* ldrGetRenderPart(LdrLoaderHDL loader, LdrPartID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->getRenderPartP(idx);
}

#endif

/*
0 // some known pathological parts to test robustness

1 10 0 0 0 1 0 0 0 1 0 0 0 1 32530.dat
1 10 0 40 0 1 0 0 0 1 0 0 0 1 4315.dat
1 10 0 80 0 1 0 0 0 1 0 0 0 1 3828.dat
1 10 0 120 0 1 0 0 0 1 0 0 0 1 3641.dat
1 10 0 160 0 1 0 0 0 1 0 0 0 1 s\41770s01.dat
1 10 0 200 0 1 0 0 0 1 0 0 0 1 32278.dat
1 10 0 240 0 1 0 0 0 1 0 0 0 1 32062.dat
1 10 0 280 0 1 0 0 0 1 0 0 0 1 64782.dat
1 10 0 360 0 1 0 0 0 1 0 0 0 1 92908.dat
1 10 0 480 0 1 0 0 0 1 0 0 0 1 3479.dat
1 10 0 580 0 1 0 0 0 1 0 0 0 1 4079.dat
0 10 0 0 0 1 0 0 0 1 0 0 0 1 4791.dat
1 10 0 0 0 1 0 0 0 1 0 0 0 1 t1044.dat
*/
