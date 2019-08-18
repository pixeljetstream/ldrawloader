/**
 * Copyright (c) 2019 Christoph Kubisch
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

const float Loader::NO_AREA_TRIANGLE_DOT = 0.9999f;
const float Loader::FORCED_HARD_EDGE_DOT = 0.2f;
const float Loader::CHAMFER_PARALLEL_DOT = 0.999f;
const float Loader::ANGLE_45_DOT = 0.7071f;

template <class TvtxIndex_t, class TvtxIndexPair, int VTX_BITS, int VTX_TRIS>
class TMesh
{
public:
  typedef TvtxIndex_t vtxIndex_t;

  static const vtxIndex_t INVALID_VTX = vtxIndex_t(~0);
  static const uint32_t   INVALID     = uint32_t(~0);

  typedef TvtxIndexPair vtxPair_t;

  static vtxPair_t make_vtxPair(vtxIndex_t vtxA, vtxIndex_t vtxB)
  {
    if(vtxA < vtxB) {
      return vtxPair_t(vtxA) | (vtxPair_t(vtxB) << VTX_BITS);
    }
    else {
      return vtxPair_t(vtxB) | (vtxPair_t(vtxA) << VTX_BITS);
    }
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

  // additional left/right for non-manifold overflow
  // we currently assume there is never more than 4 triangles connected to the same edge
  struct EdgeNM
  {
    uint32_t triLeft;
    uint32_t triRight;
  };

  struct Edge
  {
    // left triangle has its edge running from A to B
    // right triangle in opposite direction.

    vtxIndex_t vtxA;
    vtxIndex_t vtxB;
    uint32_t   triLeft;
    uint32_t   triRight;
    uint32_t   flag;

    bool isDead() const { return triLeft == INVALID; }

    bool isOpen() const { return triRight == INVALID; }

    bool hasFace(uint32_t idx) const { return triLeft == idx || triRight == idx; }

    vtxIndex_t otherVertex(vtxIndex_t idx) const { return idx != vtxA ? vtxA : vtxB; }

    uint32_t otherTri(uint32_t idx) const { return idx != triLeft ? triLeft : triRight; }

    uint32_t getTri(uint32_t right) const { return right ? triRight : triLeft; }

    vtxIndex_t getVertex(uint32_t b) const { return b ? vtxB : vtxA; }

    vtxPair_t getPair() const { return make_vtxPair(vtxA, vtxB); }
  };

  uint32_t numVertices  = 0;
  uint32_t numTriangles = 0;
  uint32_t numEdges     = 0;

  vtxIndex_t* triangles = nullptr;

  Connectivity vtxTriangles;
  Connectivity vtxEdges;

  Loader::BitArray                        triAlive;
  Loader::TVector<Edge>                   edges;
  Loader::TVector<EdgeNM>                 edgesNM;
  std::unordered_map<vtxPair_t, uint32_t> lookupEdge;

  inline bool resizeVertices(uint32_t num)
  {
    numVertices = num;
    vtxEdges.infos.resize(numVertices);

    if(VTX_TRIS) {
      vtxTriangles.infos.resize(numVertices);
    }

    if(num > (1 << VTX_BITS)) {
      // FIXME check against this error
      fprintf(stderr, "Mesh VTX_BITS too small\n");
      exit(-1);
      return true;
    }

    return false;
  }

  inline bool getVertexClosable(vtxIndex_t vertex)
  {
    uint32_t        open = 0;
    uint32_t        count;
    const uint32_t* curEdges = vtxEdges.getConnected(vertex, count);
    for(uint32_t e = 0; e < count; e++) {
      open += edges[curEdges[e]].isOpen() && !edges[curEdges[e]].isDead();
    }
    return open == 2;
  }

  inline Edge* getEdge(vtxIndex_t vtxA, vtxIndex_t vtxB)
  {
    const auto it = lookupEdge.find(make_vtxPair(vtxA, vtxB));
    if(it != lookupEdge.cend()) {
      return &edges[it->second];
    }
    return nullptr;
  }

  inline uint32_t getEdgeIdx(vtxIndex_t vtxA, vtxIndex_t vtxB)
  {
    const auto it = lookupEdge.find(make_vtxPair(vtxA, vtxB));
    if(it != lookupEdge.cend()) {
      return it->second;
    }
    return INVALID;
  }

  inline vtxIndex_t* getTriangle(uint32_t t) { return &triangles[t * 3]; }

  inline const vtxIndex_t* getTriangle(uint32_t t) const { return &triangles[t * 3]; }

  inline LdrVector getTriangleNormal(uint32_t t, const LdrVector* positions) const
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    return vec_normalize(vec_cross(vec_sub(positions[idxB], positions[idxA]), vec_sub(positions[idxC], positions[idxA])));
  }

  inline const vtxIndex_t getTriangleOtherVertex(uint32_t t, const Edge& edge) const
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

    return INVALID_VTX;
  }

  inline uint32_t findTriangleVertex(uint32_t t, vtxIndex_t vtx) const
  {
    const vtxIndex_t* indices = getTriangle(t);
    if(indices[0] == vtx)
      return 0;
    if(indices[1] == vtx)
      return 1;
    if(indices[2] == vtx)
      return 2;
    return INVALID;
  }

  inline void replaceTriangleVertex(uint32_t t, vtxIndex_t vtx, vtxIndex_t newVtx)
  {
    vtxIndex_t* indices = getTriangle(t);
    if(indices[0] == vtx)
      indices[0] = newVtx;
    if(indices[1] == vtx)
      indices[1] = newVtx;
    if(indices[2] == vtx)
      indices[2] = newVtx;
  }

  // returns sub index within vertex edge list
  inline uint32_t getNextVertexSubEdge(vtxIndex_t vertex, uint32_t subIndex, uint32_t triIndex, uint32_t triIndexNot) const
  {
    if(triIndex == INVALID)
      return subIndex;

    uint32_t        count;
    const uint32_t* curEdges = vtxEdges.getConnected(vertex, count);
    for(uint32_t i = 0; i < count; i++) {
      uint32_t    edgeIndex = curEdges[i];
      const Edge& edge      = edges[edgeIndex];

      if(i == subIndex || edge.isDead()) {
        continue;
      }

      if(edge.hasFace(triIndex) && (edge.isOpen() || edge.otherTri(triIndex) != triIndexNot)) {
        return i;
      }
    }
    return subIndex;
  }

  // returns edge index
  inline uint32_t getNextVertexEdge(vtxIndex_t vertex, uint32_t edgePrevIndex, uint32_t triIndex, uint32_t triIndexNot) const
  {
    if(triIndex == INVALID)
      return edgePrevIndex;

    uint32_t        count;
    const uint32_t* curEdges = vtxEdges.getConnected(vertex, count);
    for(uint32_t i = 0; i < count; i++) {
      uint32_t    edgeIndex = curEdges[i];
      const Edge& edge      = edges[edgeIndex];

      if(edgeIndex == edgePrevIndex || edge.isDead()) {
        continue;
      }

      if(edge.hasFace(triIndex) && (edge.isOpen() || edge.otherTri(triIndex) != triIndexNot)) {
        return edgeIndex;
      }
    }
    return edgePrevIndex;
  }

  uint32_t addEdge(vtxIndex_t vtxA, vtxIndex_t vtxB, uint32_t tri, uint32_t& nonManifold)
  {
    vtxPair_t pair = make_vtxPair(vtxA, vtxB);
    auto      it   = lookupEdge.find(pair);
    if(it != lookupEdge.end()) {
      Edge& edge = edges[it->second];

      if(edge.triLeft == tri || edge.triRight == tri) {
        // can happen when we process the same triangle twice
        it = it;
      }
      else if(vtxA == edge.vtxB && edge.triRight == INVALID) {
        edge.triRight = tri;
      }
      else {
        // non-manifold
        edgesNM.resize(edges.size(), {INVALID, INVALID});

        EdgeNM& edgeNM = edgesNM[it->second];
        if(edge.vtxA == vtxA && edgeNM.triLeft == INVALID) {
          edgeNM.triLeft = tri;
        }
        else if(edge.vtxB == vtxA && edgeNM.triRight == INVALID) {
          edgeNM.triRight = tri;
        }
        else {
          assert(0);
        }

        nonManifold = it->second;
      }

      return it->second;
    }
    else {
      uint32_t edgeIdx = numEdges;
      edges.push_back({vtxA, vtxB, tri, INVALID, 0});
      lookupEdge.insert({pair, edgeIdx});

      vtxEdges.add(vtxA, edgeIdx);
      vtxEdges.add(vtxB, edgeIdx);

      numEdges++;
      return edgeIdx;
    }
  }

  inline void removeEdge(vtxIndex_t vtxA, vtxIndex_t vtxB, uint32_t tri)
  {
    vtxPair_t pair = make_vtxPair(vtxA, vtxB);
    auto      it   = lookupEdge.find(pair);
    if(it == lookupEdge.end())
      return;

    Edge& edge = edges[it->second];
    if(it->second >= edgesNM.size()) {
      if(edge.triRight == tri) {
        edge.triRight = INVALID;
      }
      else if(edge.triRight != INVALID) {
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

        lookupEdge.erase(pair);
      }
    }
    else {
      EdgeNM& edgeNM = edgesNM[it->second];
      if(edgeNM.triLeft == tri) {
        edgeNM.triLeft = INVALID;
      }
      else if(edgeNM.triRight == tri) {
        edgeNM.triRight = INVALID;
      }
      if(edge.triRight == tri) {
        edge.triRight   = edgeNM.triRight;
        edgeNM.triRight = INVALID;
      }
      else if(edge.triLeft == tri) {
        if(edgeNM.triLeft != INVALID) {
          edge.triLeft   = edgeNM.triLeft;
          edgeNM.triLeft = INVALID;
        }
        else if(edge.triRight != INVALID) {
          edge.triLeft  = edge.triRight;
          uint32_t temp = edge.vtxB;
          edge.vtxB     = edge.vtxA;
          edge.vtxA     = temp;
          edge.triRight = INVALID;

          temp            = edgeNM.triLeft;
          edgeNM.triLeft  = edgeNM.triRight;
          edgeNM.triRight = temp;
        }
        else {
          edge.triLeft = INVALID;

          vtxEdges.remove(edge.vtxA, it->second);
          vtxEdges.remove(edge.vtxB, it->second);

          lookupEdge.erase(pair);
        }
      }
    }
  }

  inline uint32_t isEdgeClosed(vtxIndex_t vtxA, vtxIndex_t vtxB) const
  {
    vtxPair_t pair = make_vtxPair(vtxA, vtxB);
    auto      it   = lookupEdge.find(pair);
    if(it != lookupEdge.end()) {
      const Edge& edge = edges[it->second];
      // the right triangle if not specified yet, must be in reverse direction
      return (edge.triRight != INVALID || edge.vtxA == vtxA) ? it->second : INVALID;
    }
    return INVALID;
  }

  inline bool checkNoArea(uint32_t t, const LdrVector* pos)
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    LdrVector vecA = pos[idxA];
    LdrVector vecB = pos[idxB];
    LdrVector vecC = pos[idxC];

    float dot = fabsf(vec_dot(vec_normalize(vec_sub(vecB, vecA)), vec_normalize(vec_sub(vecC, vecA))));
    return dot > Loader::NO_AREA_TRIANGLE_DOT;
  }

  inline uint32_t checkNonManifoldTriangle(uint32_t t, uint32_t vtx[2])
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    uint32_t edge = isEdgeClosed(idxA, idxB);
    if(edge != INVALID) {
      vtx[0] = idxA;
      vtx[1] = idxB;
      return edge;
    }
    edge = isEdgeClosed(idxB, idxC);
    if(edge != INVALID) {
      vtx[0] = idxB;
      vtx[1] = idxC;
      return edge;
    }
    edge = isEdgeClosed(idxC, idxA);
    if(edge != INVALID) {
      vtx[0] = idxC;
      vtx[1] = idxA;
      return edge;
    }

    return INVALID;
  }

  inline uint32_t checkNonManifoldQuad(uint32_t t, uint32_t vtx[2])
  {
    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];
    uint32_t idxD = triangles[t * 3 + 4];

    uint32_t edge = isEdgeClosed(idxA, idxB);
    if(edge != INVALID) {
      vtx[0] = idxA;
      vtx[1] = idxB;
      return edge;
    }
    edge = isEdgeClosed(idxB, idxC);
    if(edge != INVALID) {
      vtx[0] = idxB;
      vtx[1] = idxC;
      return edge;
    }
    edge = isEdgeClosed(idxC, idxD);
    if(edge != INVALID) {
      vtx[0] = idxC;
      vtx[1] = idxD;
      return edge;
    }
    edge = isEdgeClosed(idxD, idxA);
    if(edge != INVALID) {
      vtx[0] = idxD;
      vtx[1] = idxA;
      return edge;
    }

    return INVALID;
  }

  inline uint32_t addTriangle(uint32_t t)
  {
    uint32_t nonManifold = INVALID;

    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    if(idxA == idxB || idxB == idxC || idxA == idxC)
      return INVALID;

    if(VTX_TRIS) {
      vtxTriangles.add(idxA, t);
      vtxTriangles.add(idxB, t);
      vtxTriangles.add(idxC, t);
    }

    addEdge(idxA, idxB, t, nonManifold);
    addEdge(idxB, idxC, t, nonManifold);
    addEdge(idxC, idxA, t, nonManifold);

    if(triAlive.size() <= t) {
      triAlive.resize(t + 1, false);
      numTriangles = t + 1;
    }

    assert(!triAlive.getBit(t));
    triAlive.setBit(t, true);

    return nonManifold;
  }

  inline void removeTriangle(uint32_t t)
  {
    if(!triAlive.getBit(t))
      return;

    uint32_t idxA = triangles[t * 3 + 0];
    uint32_t idxB = triangles[t * 3 + 1];
    uint32_t idxC = triangles[t * 3 + 2];

    removeEdge(idxA, idxB, t);
    removeEdge(idxB, idxC, t);
    removeEdge(idxC, idxA, t);

    if(VTX_TRIS) {
      vtxTriangles.remove(idxA, t);
      vtxTriangles.remove(idxB, t);
      vtxTriangles.remove(idxC, t);
    }

    triAlive.setBit(t, false);
  }

  bool init(uint32_t numV, uint32_t numT, vtxIndex_t* tris)
  {
    bool nonManifold = false;

    numTriangles = numT;
    numEdges     = 0;
    triangles    = tris;

    edges.reserve(numT * 3);
    lookupEdge.reserve(numT * 3);
    triAlive.resize(numT, false);

    resizeVertices(numV);

    for(uint32_t t = 0; t < numT; t++) {
      nonManifold = (addTriangle(t) != INVALID) || nonManifold;
    }

    return nonManifold;
  }

  void reserve(uint32_t numV, uint32_t numT, vtxIndex_t* tris)
  {
    numEdges     = 0;
    numTriangles = numT;
    triangles    = tris;

    edges.reserve(numT * 3);
    lookupEdge.reserve(numT * 3);
    triAlive.resize(numT, false);

    resizeVertices(numV);
  }
};

typedef TMesh<uint32_t, uint32_t, 16, 0> Mesh;
typedef TMesh<uint32_t, uint32_t, 16, 1> MeshFull;

//////////////////////////////////////////////////////////////////////////

static const uint32_t EDGE_HARD_BIT     = 1;
static const uint32_t EDGE_OPTIONAL_BIT = 2;
static const uint32_t EDGE_HARD_FLOATER_BIT = 4;

// separate class due to potential template usage for Mesh
class MeshUtils
{
public:
  static void initNonManifold(Mesh& mesh, Loader::BuilderPart& builder)
  {
    mesh.reserve(builder.positions.size(), builder.triangles.size() / 3, builder.triangles.data());

    // fill mesh and fix non-manifold
    builder.connections.resize(builder.positions.size(), LDR_INVALID_IDX);

    uint32_t numT = builder.triangles.size() / 3;
    uint32_t numV = builder.positions.size();


    bool nonManifold = false;
    for(uint32_t t = 0; t < numT; t++) {
      uint32_t nonManifoldEdge;

      bool     isQuad = builder.quads[t] == t;
      uint32_t vtx[2];
      if(isQuad) {
        nonManifoldEdge = mesh.checkNonManifoldQuad(t, vtx);
      }
      else {
        //if (mesh.checkNoArea(t, builder.positions.data()))
        //continue;
        nonManifoldEdge = mesh.checkNonManifoldTriangle(t, vtx);
      }
      bool skip = false;
      if(nonManifoldEdge != Mesh::INVALID) {
        // test against coplanar overlapping primitives

        Mesh::Edge& edge        = mesh.edges[nonManifoldEdge];
        uint32_t    tOther      = edge.vtxA == vtx[0] ? edge.triLeft : edge.triRight;
        LdrVector   normal      = mesh.getTriangleNormal(t, builder.positions.data());
        LdrVector   normalOther = mesh.getTriangleNormal(tOther, builder.positions.data());
        if(vec_dot(normal, normalOther) > 0.99) {
          LdrVector side = vec_cross(normal, vec_sub(builder.positions[edge.vtxA], builder.positions[edge.vtxB]));

          // find furthest vertex from the edge

          uint32_t  vtx       = mesh.getTriangleOtherVertex(t, edge);
          uint32_t  vtxOther  = mesh.getTriangleOtherVertex(tOther, edge);
          LdrVector vec0      = vec_sub(builder.positions[vtx], builder.positions[edge.vtxA]);
          LdrVector vec1      = vec_sub(builder.positions[vtx], builder.positions[edge.vtxB]);
          LdrVector vecOther0 = vec_sub(builder.positions[vtxOther], builder.positions[edge.vtxA]);
          LdrVector vecOther1 = vec_sub(builder.positions[vtxOther], builder.positions[edge.vtxB]);
          LdrVector vec       = vec_sq_length(vec0) > vec_sq_length(vec1) ? vec0 : vec1;
          LdrVector vecOther  = vec_sq_length(vecOther0) > vec_sq_length(vecOther1) ? vecOther0 : vecOther1;

          float dot      = vec_dot(vec, side);
          float dotOther = vec_dot(vecOther, side);

          // both non-edge vertices are on the same side, i.e. primitives overlap
          if(dot < 0 == dotOther < 0) {
            // pseudo heuristic who is more important to be kept
            // we assume that "bigger" (more distant vertices) are more important
            if(vec_sq_length(vec) < vec_sq_length(vecOther)) {
              // skip this triangle/quad
              skip = true;
              t += isQuad ? 1 : 0;
            }
            else {
              // remove the previous triangle/quad
              if(builder.isQuad(tOther)) {
                // the other triangle
                mesh.removeTriangle(builder.quads[tOther] == tOther ? tOther + 1 : builder.quads[tOther]);
              }
              mesh.removeTriangle(tOther);
            }
            nonManifoldEdge = Mesh::INVALID;
          }
        }
      }
      if(!skip) {
        nonManifoldEdge = mesh.addTriangle(t);
      }

      nonManifold = nonManifold || (nonManifoldEdge != Mesh::INVALID);
    }

    if(!nonManifold)
      return;

    // if nonManifold remain we test against two scenarios per edge
    //
    // view along edge x:
    //
    // 3 surfaces, we keep those two triangles connected that came first
    //
    // tri 0
    //     \
    //      x __ tri 2  (2 is split away)
    //      |
    //  tri 1
    //
    // 4 surfaces, we try to split x on only one side, if it make sense from the surrounding
    //             topology. Otherwise we remove triangle 2 & 3.
    //
    //       0
    //       |
    //   1 __x__ 2
    //       |
    //       3


    Loader::TVector<LdrVector> triNormals;
    Loader::TVector<uint32_t>  triTemps;
    Loader::TVector<uint32_t>  triDelete;

    triTemps.reserve(32);
    triNormals.reserve(32);

    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      Mesh::Edge edge = mesh.edges[e];
      if(e >= mesh.edgesNM.size()) {
        break;
      }

      Mesh::EdgeNM edgeNM = mesh.edgesNM[e];

      if(edgeNM.triRight != Mesh::INVALID && edgeNM.triLeft != Mesh::INVALID) {
        // 4 surfaces per edge
        // We may need to split one or two vertices, find out which by clustering the triangles/edges
        // into vtxA and vtxB connections.

        // We prefer to split the vertex at the opposite side of the average normal (see normal dot later)

        uint32_t edgeTriangles[4] = {edge.triLeft, edge.triRight, edgeNM.triLeft, edgeNM.triRight};

        LdrVector vecAB = vec_sub(builder.positions[edge.vtxB], builder.positions[edge.vtxA]);
        LdrVector normalAB = vec_normalize(vecAB);

        LdrVector normal[2] = {{0, 0, 0}, {0, 0, 0}};

        // index into edgeTriangles, gives us one triangle that is connected to the A or B side
        uint32_t connectedSubTris[2];
        // edge that is connected to this sub-triangle on the A or B side
        uint32_t connectedEdge[2][4];

        for(uint32_t side = 0; side < 2; side++) {
          uint32_t        edgeCount;
          const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(edge.getVertex(side), edgeCount);
          for(uint32_t ev = 0; ev < edgeCount; ev++) {
            // skip non-manifold edge
            if(edgeIndices[ev] == e)
              continue;

            bool     foundLeft  = false;
            bool     foundRight = false;
            uint32_t triLeft    = mesh.edges[edgeIndices[ev]].triLeft;
            uint32_t triRight   = mesh.edges[edgeIndices[ev]].triRight;
            for(uint32_t s = 0; s < 4; s++) {
              if(triLeft == edgeTriangles[s]) {
                connectedSubTris[side] = s;
                connectedEdge[side][s] = edgeIndices[ev];
              }
              if(triRight == edgeTriangles[s]) {
                connectedSubTris[side] = s;
                connectedEdge[side][s] = edgeIndices[ev];
              }
            }

            // add only triangle normals that are roughly perpendicular to the edge
            // the second heuristic below depends on detecting clusters

            LdrVector normalLeft = mesh.getTriangleNormal(triLeft, builder.positions.data());
            if (fabs(vec_dot(normalLeft,normalAB)) > Loader::ANGLE_45_DOT){
              normal[side] = vec_add(normal[side], normalLeft);
            }
            
            if(triRight != Mesh::INVALID) {
              LdrVector normalRight = mesh.getTriangleNormal(triLeft, builder.positions.data());
              if (fabs(vec_dot(normalRight,normalAB)) > Loader::ANGLE_45_DOT){
                normal[side] = vec_add(normal[side], normalRight);
              }
            }
          }
        }

        // top and bottom cluster point roughly in same direction

        if(vec_dot(vec_normalize(normal[0]), vec_normalize(normal[1])) > Loader::ANGLE_45_DOT) {
          /*
            "top down" view 
            kept vertex is tip of normal direction, we split the "lower" vertex

            splitNew ____x
                   |1\ L |         \ is non-manifold edge , L/R is left/right status 
                   |R \  |                                  (could be flipped as well, but pairings are consistent)
                   x__ kept ___x
                         | \ R |       we pick a edge-triangle connected to a non-edge triangle, connected to kept
                         |L \ 0|       that triangle (0) keeps the original vertex, the opposite triangle (1) has same L/R status
                         x__ split     and gets the split vertex.
                       
          */


          // split one vertex only
          // the one that is opposite side of normal
          uint32_t keptSide  = 1;
          uint32_t splitSide = 0;
          uint32_t splitVtx  = edge.vtxA;
          uint32_t keptVtx   = edge.vtxB;
          if(vec_dot(vecAB, normal[0]) < 0) {
            keptVtx   = edge.vtxA;
            splitVtx  = edge.vtxB;
            splitSide = 1;
            keptSide  = 0;
          }
          uint32_t splitNew = builder.addConnection(splitVtx);
          mesh.resizeVertices(builder.positions.size());

          // the opposing triangle (same left/right status will get the new vertex
          uint32_t opposite     = (connectedSubTris[keptSide] + 2) % 4;
          uint32_t triOpposite  = edgeTriangles[opposite];
          uint32_t triOpposite2 = Mesh::INVALID;

          // find all triangles connected to opposite within "lower" list
          // and add them to temp
          {
            uint32_t edgeCur = connectedEdge[splitSide][opposite];
            uint32_t edgeNext;
            uint32_t triStart = mesh.edges[edgeCur].otherTri(triOpposite);
            uint32_t triOther = triOpposite;

            while((edgeNext = mesh.getNextVertexEdge(splitVtx, edgeCur, triStart, triOther)) != edgeCur) {
              const Mesh::Edge& edgeIter = mesh.edges[edgeNext];
              triOther                   = triStart;
              triStart                   = edgeIter.otherTri(triStart);
              if(edgeNext == e) {
                triOpposite2 = triOther;
                break;
              }
              triTemps.push_back(triOther);
              edgeCur = edgeNext;
              if(edgeNext == connectedEdge[splitSide][opposite + 1] || edgeNext == connectedEdge[splitSide][(opposite + 3) % 4]) {
                triOpposite2 = triStart;
                break;
              }
            }
          }

          assert(triOpposite2 != Mesh::INVALID);

          // rebuild local triangles
          for(uint32_t s = 0; s < 4; s++) {
            mesh.removeTriangle(edgeTriangles[s]);
          }
          mesh.replaceTriangleVertex(triOpposite, splitVtx, splitNew);
          mesh.replaceTriangleVertex(triOpposite2, splitVtx, splitNew);

          for(uint32_t t = 0; t < (uint32_t)triTemps.size(); t++) {
            mesh.removeTriangle(triTemps[t]);
            mesh.replaceTriangleVertex(triTemps[t], splitVtx, splitNew);
          }
          for(uint32_t s = 0; s < 4; s++) {
            mesh.addTriangle(edgeTriangles[s]);
          }
          for(uint32_t t = 0; t < (uint32_t)triTemps.size(); t++) {
            mesh.addTriangle(triTemps[t]);
          }
        }
        else {
          for(uint32_t s = 0; s < 2; s++) {
            //mesh.removeTriangle(edgeTriangles[2+s]);
            triDelete.push_back(edgeTriangles[2 + s]);
          }
          // split both, NYI
          //assert(0);
          builder.flag.fixErrors = 1;
        }
      }
      else if(edgeNM.triLeft != Mesh::INVALID || edgeNM.triRight != Mesh::INVALID) {
        uint32_t triUsed = edgeNM.triRight != Mesh::INVALID ? edgeNM.triRight : edgeNM.triLeft;
        // 3 surfaces per edge
        // split edge
        uint32_t newA = builder.addConnection(edge.vtxA);
        uint32_t newB = builder.addConnection(edge.vtxB);
        mesh.resizeVertices(builder.positions.size());

        triTemps.push_back(triUsed);

        for(uint32_t side = 0; side < 2; side++) {
          // iterate each vertex
          uint32_t v      = edge.getVertex(side);
          uint32_t eStart = e;
          uint32_t eCur   = eStart;
          uint32_t eNext;
          uint32_t tStart = triUsed;
          uint32_t tOther = 0;
          while((eNext = mesh.getNextVertexEdge(v, eCur, tStart, tOther)) != eCur) {
            tOther = tStart;
            tStart = mesh.edges[eNext].otherTri(tStart);
            eCur   = eNext;
            if(eCur == eStart)
              return;
            if(tStart != Mesh::INVALID) {
              triTemps.push_back(tStart);
            }
          }
        }

        for(uint32_t t = 0; t < (uint32_t)triTemps.size(); t++) {
          mesh.removeTriangle(triTemps[t]);
          mesh.replaceTriangleVertex(triTemps[t], edge.vtxA, newA);
          mesh.replaceTriangleVertex(triTemps[t], edge.vtxB, newB);
        }
        for(uint32_t t = 0; t < (uint32_t)triTemps.size(); t++) {
          if(!mesh.triAlive.getBit(t)) {
            mesh.addTriangle(triTemps[t]);
          }
        }

        edgeNM.triLeft = Mesh::INVALID;
      }

      triTemps.clear();
    }

    for(size_t t = 0; t < triDelete.size(); t++) {
      mesh.removeTriangle(triDelete[t]);
    }
  }

  static void removeDeleted(Mesh& mesh, Loader::BuilderPart& builder)
  {
    uint32_t write = 0;
    for(uint32_t t = 0; t < mesh.numTriangles; t++) {
      if(mesh.triAlive.getBit(t)) {
        builder.triangles[write * 3 + 0] = builder.triangles[t * 3 + 0];
        builder.triangles[write * 3 + 1] = builder.triangles[t * 3 + 1];
        builder.triangles[write * 3 + 2] = builder.triangles[t * 3 + 2];
        builder.materials[write]         = builder.materials[t];
        write++;
      }
    }

    builder.triangles.resize(write * 3);
    builder.materials.resize(write);
  }

  static void storeLines(Mesh& mesh, Loader::BuilderPart& builder, bool optional)
  {
    Loader::TVector<LdrVertexIndex>& lines = optional ? builder.optional_lines : builder.lines;
    uint32_t                         flag  = optional ? EDGE_OPTIONAL_BIT : EDGE_HARD_BIT;

    lines.clear();
    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      const Mesh::Edge& edge = mesh.edges[e];
      if(!edge.isDead() && edge.flag & flag) {
        lines.push_back(edge.vtxA);
        lines.push_back(edge.vtxB);
      }
    }
  }
  static void initEdgeLines(Mesh& mesh, Loader::BuilderPart& builder, bool optional)
  {
    Loader::TVector<uint32_t> path;
    path.reserve(16);

    Loader::TVector<LdrVertexIndex>& lines = optional ? builder.optional_lines : builder.lines;
    uint32_t                         flag  = optional ? EDGE_OPTIONAL_BIT : EDGE_HARD_BIT;

    // flag mesh edges as lines
    // Some fixing required due to floaters (lines are not actually existing as triangle edges)
    // or due to non-manifold fix before.

    uint32_t numOrigLines = lines.size() / 2;

    for(uint32_t i = 0; i < lines.size() / 2; i++) {
      uint32_t lineA = lines[i * 2 + 0];
      uint32_t lineB = lines[i * 2 + 1];

      uint32_t otherA = builder.connections[lineA];
      uint32_t otherB = builder.connections[lineB];

      // both vertices were duplicated, then also duplicate this line once
      if(i < numOrigLines && (otherA != LDR_INVALID_IDX || otherB != LDR_INVALID_IDX)) {
        if(otherA != LDR_INVALID_IDX && otherB != LDR_INVALID_IDX) {
          lines.push_back(otherA);
          lines.push_back(otherB);
        }
        else if(otherA != LDR_INVALID_IDX) {
          lines.push_back(otherA);
          lines.push_back(lineB);
        }
        else if(otherB != LDR_INVALID_IDX) {
          lines.push_back(lineA);
          lines.push_back(otherB);
        }
      }

      Mesh::Edge* edgeFound = mesh.getEdge(lineA, lineB);
      if(edgeFound) {
        edgeFound->flag |= flag;
      }
      else {
        // floating edges (no triangles) are ugly, try to find path between them
        // once from both directions in case there is t-junction

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
            uint32_t vtests[2] = {v, builder.connections[v]};
            for(uint32_t t = 0; t < 2; t++) {
              uint32_t vtest = vtests[t];
              if(vtest == LDR_INVALID_IDX)
                continue;

              uint32_t        edgeCount;
              const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(vtest, edgeCount);
              for(uint32_t e = 0; e < edgeCount; e++) {
                const Mesh::Edge& edge = mesh.edges[edgeIndices[e]];
                assert(!edge.isDead());
                Mesh::vtxIndex_t vOther = edge.otherVertex(vtest);
                LdrVector        vecCur = vec_normalize(vec_sub(builder.positions[vOther], builder.positions[vtest]));
                float            dist   = vec_sq_length(vec_sub(builder.positions[vOther], builder.positions[vEnd]));
                float            dot    = vec_dot(vecCur, vecEdge);
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
          }
        }
      }
    }
    lines.clear();
  }


  static bool fixTjunctions(Mesh& mesh, Loader::BuilderPart& builder)
  {
    bool modified = false;

    // removes t-junctions and closable gaps
    builder.flag.canChamfer = 1;

    uint32_t numVertices = (uint32_t)builder.positions.size();

    Loader::BitArray          processed(mesh.edges.size(), false);

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

        uint32_t         triOld = edgeA.triLeft;

        LdrVector        posA = builder.positions[v];
        Mesh::vtxIndex_t vEnd = edgeA.otherVertex(v);

        float     lengthA;
        LdrVector vecA = vec_normalize_length(vec_sub(builder.positions[vEnd], posA), lengthA);

        std::vector<uint32_t> path;
        path.reserve(128);

        Mesh::vtxIndex_t vNext       = v;
        uint32_t         edgeIdxSkip = edgeIdxA;
        uint32_t         edgeNext    = 0;
        bool             singleTriangle = true;

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

            Mesh::vtxIndex_t vC = edgeC.otherVertex(vNext);

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
        if (singleTriangle) {
          continue;
        }

        modified = true;

        processed.setBit(edgeIdxA, true);

        Mesh::vtxIndex_t triIndices[3];
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
        uint32_t flagACorner = mesh.getEdge(edgeA.vtxA,vCorner)->flag;
        uint32_t flagBCorner = mesh.getEdge(edgeA.vtxB,vCorner)->flag;

        mesh.removeTriangle(triOld);

        uint32_t quadOld = builder.quads[triOld];
        if (quadOld != LDR_INVALID_IDX){
          builder.quads[quadOld] = LDR_INVALID_IDX;
          builder.quads[triOld] = LDR_INVALID_IDX;
        }

        // FIXME if triOld is quad, then try to distribute new edges on both opposite vertices (A,B)
        //
        //    A___B
        //    |\ /|
        //    x_x_x
        //
        //    instead of all new triangles within a single triangle only (only A)
        //
        //    A___B
        //    |\  |
        //    |\\ |
        //    x_x_x

        uint32_t flagPath = edgeA.flag;
        Mesh::vtxIndex_t vFirst = edgeA.vtxA;

        for(size_t i = 0; i < path.size(); i++) {
          uint32_t    edgeIdxC = path[i];
          Mesh::Edge& edgeC    = mesh.edges[edgeIdxC];
          processed.setBit(edgeIdxC, true);
          // inherit edge flag
          edgeC.flag |= flagPath;

          // make new triangles, first re-uses old slot
          edgeC.triRight = i == 0 ? triOld : builder.triangles.size() / 3;

          uint32_t triIdx;
          if(i != 0) {
            triIdx = uint32_t(builder.triangles.size() / 3);
            builder.triangles.push_back(vFirst);
            builder.triangles.push_back(edgeC.otherVertex(vFirst));
            builder.triangles.push_back(vCorner);
            builder.materials.push_back(builder.materials[triOld]);
          }
          else {
            triIdx                            = triOld;
            builder.triangles[triOld * 3 + 0] = vFirst;
            builder.triangles[triOld * 3 + 1] = edgeC.otherVertex(vFirst);
            builder.triangles[triOld * 3 + 2] = vCorner;
          }

          mesh.triangles = builder.triangles.data();

          uint32_t nonManifold = mesh.addTriangle(triIdx);
          if(nonManifold != Mesh::INVALID) {
            builder.flag.canChamfer = 0;
          }

          vFirst = edgeC.otherVertex(vFirst);
        }

        // re-apply edge flag
        Mesh::Edge* edgeACorner =  mesh.getEdge(edgeA.vtxA,vCorner);
        Mesh::Edge* edgeBCorner =  mesh.getEdge(edgeA.vtxB,vCorner);
        if (edgeACorner) edgeACorner->flag = flagACorner;
        if (edgeBCorner) edgeBCorner->flag = flagBCorner;

        processed.resize(mesh.edges.size(), false);
        respin = true;
      }
      if(respin) {
        v = v - 1;
      }
    }

    return modified;
  }

  static void fixBuilderPart(Loader::BuilderPart& builder, const Loader::Config& config)
  {
    Mesh mesh;
    MeshUtils::initNonManifold(mesh, builder);
    MeshUtils::initEdgeLines(mesh, builder, false);
    MeshUtils::initEdgeLines(mesh, builder, true);
    if(config.partFixTjunctions){
      MeshUtils::fixTjunctions(mesh, builder);
    }
    MeshUtils::storeLines(mesh, builder, false);
    MeshUtils::storeLines(mesh, builder, true);
    MeshUtils::removeDeleted(mesh, builder);
  }

  static void chamferRenderPart(MeshFull& mesh, Loader::BuilderRenderPart& builder, const LdrPart& part, const float chamferPreferred)
  {
    // copy over original triangles first
    Loader::TVector<uint32_t> smoothedTriangles = builder.triangles;
    builder.trianglesC                          = builder.triangles;
    builder.materialsC.resize(part.flag.hasComplexMaterial ? part.num_triangles : 0);

    bool hasMaterials = !builder.materialsC.empty();

    if(hasMaterials) {
      memcpy(builder.materialsC.data(), part.materials, sizeof(LdrMaterialID) * part.num_triangles);
    }

    // all vertices that were split before to account for hard edges are relevant here
    // that means we don't chamfer open-edges

    // first pass is to create new chamfer vertices
    // one vertex per every split vertex

    // store where the chamfered begin
    Loader::TVector<uint32_t> vtxChamferBegin(part.num_positions, 0);

    for(uint32_t v = 0; v < part.num_positions; v++) {
      uint32_t outCount = builder.vtxOutCount[v];
      if(outCount > 1) {
        vtxChamferBegin[v] = (uint32_t)builder.vertices.size();

        // find minimum chamfer length (distances per triangle cluster would be safer)
        uint32_t        edgeCount;
        const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(v, edgeCount);

        float dist = FLT_MAX;

        LdrVector avgNormal = {0, 0, 0};
        LdrVector avgPos    = {0, 0, 0};

        for(uint32_t e = 0; e < edgeCount; e++) {
          const MeshFull::Edge edge = mesh.edges[edgeIndices[e]];
          dist = std::min(dist, vec_sq_length(vec_sub(part.positions[edge.vtxA], part.positions[edge.vtxB])));
        }
        const float chamferDistance = std::min(chamferPreferred, dist * 0.40f);

        uint32_t materialTri;

        // create new output vertex that is chamfered
        for(uint32_t o = 0; o < outCount; o++) {
          uint32_t outIdx = builder.vtxOutBegin[v] + o;

          LdrVector normal = builder.vertices[outIdx].normal;

          const Loader::BuilderRenderPart::EdgePair& edgePair = builder.vtxOutEdgePairs[outIdx];
          // get the two edges that define the triangle cluster for this output vertex
          const MeshFull::Edge& edgeA = mesh.edges[edgePair.edgeA];
          const MeshFull::Edge& edgeB = mesh.edges[edgePair.edgeB];
          uint32_t              tri   = edgePair.triA;
          materialTri                 = tri;


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
          if(!((edgeA.vtxB == v && edgeA.triLeft == edgePair.triA) || (edgeA.vtxA == v && edgeA.triRight == edgePair.triA))) {
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
          bool  isSame          = vec_dot(vecBA, vecBC) >  Loader::CHAMFER_PARALLEL_DOT;

          LdrVector shift;
          if (isSame) {
            chamferModifier = 0;
            shift = {0,0,0};
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

          // just in case things go beyound south
          if (isnan(chamferModifier) || isinf(chamferModifier)){
            chamferModifier = 0;
          }

          LdrVector vecDelta = vec_mul(shift, (chamferDistance * chamferModifier));

          if(edgeA.isOpen() || edgeB.isOpen()) {
            MeshFull::vtxIndex_t vOther  = edgeA.isOpen() ? edgeA.otherVertex(v) : edgeB.otherVertex(v);
            LdrVector            vecOpen = vec_normalize(vec_sub(part.positions[vOther], part.positions[v]));
            vecDelta                     = vec_mul(vecOpen, vec_dot(vecOpen, vecDelta));
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
            const uint32_t* indices  = &smoothedTriangles[triBegin];
            if(indices[0] == outIdx)
              builder.trianglesC[triBegin + 0] = newIdx;
            if(indices[1] == outIdx)
              builder.trianglesC[triBegin + 1] = newIdx;
            if(indices[2] == outIdx)
              builder.trianglesC[triBegin + 2] = newIdx;
          }
        }

        // create "inner triangle" based on out vertex count
        // 2 -> nothing
        // 3 -> 1 triangle
        // 4 -> 2 triangles
        // 5 -> 5 triangles with center point (NYI)

        // fixme, naive actual implementation outcount > 4 needs other logic
#if 1
        uint32_t outTriangleCount = outCount > 4 ? outCount : (outCount > 2 ? outCount - 2 : 0);
        uint32_t lastIndex        = builder.vertices.size();
        if(outTriangleCount > 2) {
          avgNormal = vec_normalize(avgNormal);
          builder.vertices.push_back({vec_mul(avgPos, 1.0f / float(outCount)), avgNormal});
        }

        for(uint32_t t = 0; t < outTriangleCount; t++) {
          uint32_t a = vtxChamferBegin[v] + t + 0;
          uint32_t b = vtxChamferBegin[v] + (t + 1) % outCount;
          uint32_t c = outTriangleCount > 2 ? lastIndex : vtxChamferBegin[v] + t + 2;

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
            builder.materialsC.push_back(part.materials[materialTri]);
          }
        }
#endif
      }
    }

    // second pass is to create new triangles for every hard edge
    for(uint32_t e = 0; e < mesh.numEdges; e++) {
      const MeshFull::Edge& edge = mesh.edges[e];

      if(!(edge.flag & EDGE_HARD_BIT) || edge.isOpen())
        continue;

      // find the right out vertex for this edge
      auto findChamferVertex = [&](uint32_t v, uint32_t tri) {
        uint32_t outCount = builder.vtxOutCount[v];
        for(uint32_t o = 0; o < outCount; o++) {
          uint32_t                                   outIdx   = builder.vtxOutBegin[v] + o;
          const Loader::BuilderRenderPart::EdgePair& edgePair = builder.vtxOutEdgePairs[outIdx];
          if(edgePair.edgeA == e && edgePair.triA == tri) {
            return vtxChamferBegin[v] + o;
          }
          if(edgePair.edgeB == e && edgePair.triB == tri) {
            return vtxChamferBegin[v] + o;
          }
        }
        assert(outCount == 1);
        return builder.vtxOutBegin[v];
      };

      // find the two vertices for left side
      uint32_t idxLeftA = findChamferVertex(edge.vtxA, edge.triLeft);
      uint32_t idxLeftB = findChamferVertex(edge.vtxB, edge.triLeft);
      // for right side
      uint32_t idxRightA = findChamferVertex(edge.vtxA, edge.triRight);
      uint32_t idxRightB = findChamferVertex(edge.vtxB, edge.triRight);

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
          builder.materialsC.push_back(part.materials[edge.triLeft]);
        }
        builder.trianglesC.push_back(a);
        builder.trianglesC.push_back(b);
        builder.trianglesC.push_back(c);
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

  static void buildRenderPart(Loader::BuilderRenderPart& builder, const LdrPart& part, const Loader::Config& config)
  {
    builder.bbox = part.bbox;
    builder.flag = part.flag;

    MeshFull mesh;
    mesh.init(part.num_positions, part.num_triangles, part.triangles);

    builder.vertices.reserve(part.num_positions + part.num_lines * 2);
    builder.lines.reserve(part.num_lines * 2);
    builder.triangles.resize(part.num_triangles * 3, LDR_INVALID_IDX);

    builder.triNormals.reserve(part.num_triangles);

    for(uint32_t t = 0; t < part.num_triangles; t++) {
      builder.triNormals.push_back(mesh.getTriangleNormal(t, part.positions));
    }

    // push "hard" lines
    for(uint32_t i = 0; i < part.num_lines; i++) {
      MeshFull::Edge* edge = mesh.getEdge(part.lines[i * 2 + 0], part.lines[i * 2 + 1]);
      if(edge) {
        edge->flag |= EDGE_HARD_BIT;
      }
    }
#if 1
    for (uint32_t e = 0; e < mesh.numEdges; e++){
      MeshFull::Edge& edge = mesh.edges[e];
      if (!edge.isOpen() && vec_dot(builder.triNormals[edge.triLeft], builder.triNormals[edge.triRight]) < Loader::FORCED_HARD_EDGE_DOT ){
        edge.flag |= EDGE_HARD_BIT;
      }
    }
#endif


    // distribute
    // find out vertex splits
    builder.vtxOutCount.resize(part.num_positions, 0);
    builder.vtxOutBegin.resize(part.num_positions, 0);
    builder.vtxOutEdgePairs.reserve(part.num_positions + part.num_lines * 2);

    // Render vertices may need to be split according to adjacent triangles.
    // Smooth vertex normals for triangles that are not separated by hard edges.

    for(uint32_t v = 0; v < part.num_positions; v++) {

      uint32_t newVertex     = (uint32_t)builder.vertices.size();
      builder.vtxOutBegin[v] = newVertex;

      LdrVector pos = part.positions[v];

      uint32_t        edgeCount;
      const uint32_t* edgeIndices = mesh.vtxEdges.getConnected(v, edgeCount);

      uint32_t        triCount;
      const uint32_t* triIndices = mesh.vtxTriangles.getConnected(v, triCount);

      // count hard edges
      uint32_t splitCount = 0;
      uint32_t openCount = 0;
      uint32_t hardFirst = LDR_INVALID_IDX;
      bool     twoSided  = false;
      for(uint32_t e = 0; e < edgeCount; e++) {
        const MeshFull::Edge& edge = mesh.edges[edgeIndices[e]];
        if(edge.flag & EDGE_HARD_BIT) {
          splitCount++;
          twoSided = twoSided || !edge.isOpen();
          if(hardFirst == LDR_INVALID_IDX) {
            hardFirst = e;
          }
        }
        else if (edge.isOpen()) {
          splitCount++;
        }
        openCount += edge.isOpen() ? 1 : 0;
      }
      uint32_t outCount = 1;
      if((twoSided || openCount != splitCount || openCount > 2) && splitCount > 1) {

        // There is multiple clusters separated by hard or open edges.
        // e.g
        //     \
        //      \ cluster 0
        //     2 v _____ (hard edge)
        //      / 1
        //     /
        //
#ifdef _DEBUG
        uint32_t       localEdgeIndices[256] = {};
        MeshFull::Edge localEdges[256]       = {};
#endif
        bool available[256 * 2] = {};
        for(uint32_t e = 0; e < edgeCount; e++) {
          const MeshFull::Edge& edge = mesh.edges[edgeIndices[e]];
          available[e * 2 + 0]       = edge.triLeft != MeshFull::INVALID;
          available[e * 2 + 1]       = edge.triRight != MeshFull::INVALID;
#ifdef _DEBUG
          localEdges[e]       = edge;
          localEdgeIndices[e] = edgeIndices[e];
#endif
        }

        // We are done if all triangles connected to the vertex have been merged into one of the clusters.

        uint32_t eStart   = 0;
        uint32_t workLeft = triCount;
        while(workLeft) {
          LdrVector normal = {0, 0, 0};

          // Start from edges we haven't already processed. Start fresh or
          // from the last used edge (eStart) to get clusters ordered next
          // to each other.

          uint32_t startIdx;
          uint32_t startRight = 0;
          for(uint32_t e = 0; e < edgeCount * 2; e++) {
            uint32_t eWrapped = (eStart * 2 + e) % (edgeCount * 2);
            if(available[eWrapped]) {
              startIdx   = eWrapped / 2;
              startRight = eWrapped % 2;
              break;
            }
          }

          const MeshFull::Edge& startEdge = mesh.edges[edgeIndices[startIdx]];
          uint32_t              startTri  = startEdge.getTri(startRight);
          uint32_t              lastTri;

          auto mergeTriangle = [&](uint32_t edgeIdx, uint32_t edgeRight, uint32_t tri) {
            // add adjacent triangle
            builder.triangles[tri * 3 + mesh.findTriangleVertex(tri, v)] = newVertex;
            normal                                                       = vec_add(normal, builder.triNormals[tri]);
            available[edgeIdx * 2 + edgeRight]                           = false;
            lastTri                                                      = tri;
            workLeft--;
          };

          // iterate edges around v
          // The edges are stored in random ordering, so we need to derive
          // edges connected via a given triangle.

          auto iterateVertexEdges = [&](uint32_t edgeIdx, uint32_t tri, uint32_t triOther) {
            uint32_t nextIdx;
            while((nextIdx = mesh.getNextVertexSubEdge(v, edgeIdx, tri, triOther)) != edgeIdx) {
              const MeshFull::Edge& edgeNext                              = mesh.edges[edgeIndices[nextIdx]];
              available[nextIdx * 2 + (edgeNext.triRight == tri ? 1 : 0)] = false;

              tri = edgeNext.otherTri(tri);
              if(edgeNext.flag & EDGE_HARD_BIT || tri == MeshFull::INVALID) {
                return nextIdx;
              }
              else {
                triOther = edgeNext.otherTri(tri);
                edgeIdx  = nextIdx;

                mergeTriangle(edgeIdx, (edgeNext.triRight == tri ? 1 : 0), tri);
              }
            }
            return nextIdx;
          };

          // when we start from a random edge, it could be an edge within a cluster
          //
          //  \     cluster (spans two triangles)
          //   \         /
          //    \       .
          //     \ Tri / (start edge, not hard)
          //      \ A .
          //       \ /  Tri B
          //        v ________ (hard edge)
          //
          // That's why we walk into two directions from the start to hit another hard/open edge.

          // first direction
          Loader::BuilderRenderPart::EdgePair pair;

          mergeTriangle(startIdx, startRight, startTri);
          uint32_t eB = iterateVertexEdges(startIdx, startTri, startEdge.otherTri(startTri));
          pair.edgeB  = edgeIndices[eB];
          pair.triB   = lastTri;

          lastTri = startTri;

          // other direction if available and we are not a hard edge
          startTri = startEdge.otherTri(startTri);
          if(startEdge.flag & EDGE_HARD_BIT) {
            startTri = MeshFull::INVALID;
          }
          else if(startTri != MeshFull::INVALID) {
            mergeTriangle(startIdx, startRight ^ 1, startTri);
          }

          uint32_t eA = iterateVertexEdges(startIdx, startTri, startEdge.otherTri(startTri));
          pair.edgeA  = edgeIndices[eA];
          pair.triA   = lastTri;
          builder.vtxOutEdgePairs.push_back(pair);

          eStart = eA == eStart ? eB : eA;

          builder.vertices.push_back(make_vertex(pos, normal));
          newVertex++;
        }
      }
      else {
        // single cluster
        // merge all triangles

        LdrVector normal = {0, 0, 0};

        for(uint32_t t = 0; t < triCount; t++) {
          uint32_t tri = triIndices[t];
          // average normal
          normal = vec_add(normal, builder.triNormals[tri]);
          // distribute new vertex index
          builder.triangles[tri * 3 + mesh.findTriangleVertex(tri, v)] = newVertex;
        }

        builder.vtxOutEdgePairs.push_back({});

        builder.vertices.push_back(make_vertex(pos, normal));
        newVertex++;
      }

      builder.vtxOutCount[v] = newVertex - builder.vtxOutBegin[v];
    }

    // for lines pick first out vertex
    for(uint32_t i = 0; i < part.num_lines; i++) {
      builder.lines.push_back(builder.vtxOutBegin[part.lines[i * 2 + 0]]);
      builder.lines.push_back(builder.vtxOutBegin[part.lines[i * 2 + 1]]);
    }

    if(config.partFixTjunctions && config.renderpartChamfer && part.flag.canChamfer) {
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

#if LDR_CFG_THREADSAFE_RESOLVES
  m_finishedPartLoad.resize(MAX_PARTS, false);
  m_finishedPrimitiveLoad.resize(MAX_PRIMS, false);
  m_finishedPartFix.resize(MAX_PARTS, false);
  m_finishedPartRenderBuild.resize(MAX_PARTS, false);
#endif

  m_startedPartLoad.resize(MAX_PARTS, false);
  m_startedPrimitiveLoad.resize(MAX_PRIMS, false);
  m_startedPartFix.resize(MAX_PARTS, false);
  m_startedPartRenderBuild.resize(MAX_PARTS, false);


  // load material config
  Text txt;

  LdrMaterialType type = LDR_MATERIAL_SOLID;

  if(txt.load((m_config.basePathString + "/LDConfig.ldr").c_str())) {
    char* line = txt.buffer;
    for(size_t i = 0; i < txt.size; i++) {
      if(txt.buffer[i] == '\r')
        txt.buffer[i] = ' ';
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
  m_startedPartFix.clear();

#if LDR_CFG_THREADSAFE_RESOLVES
  m_finishedPartLoad.clear();
  m_finishedPrimitiveLoad.clear();
  m_finishedPartRenderBuild.clear();
  m_finishedPartFix.clear();
#endif

  m_partRegistry.clear();
  m_shapeRegistry.clear();
  m_materials.clear();
}

LdrResult Loader::registerInternalPart(const char* filename, const std::string& foundname, bool isPrimitive, bool startLoad, PartEntry& entry)
{
  LdrResult result = LDR_SUCCESS;

  SpinMutex::Scoped mutex(m_partRegistryMutex);
  PartEntry         newentry;
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
        m_startedPrimitiveLoad.setBit_ts(entry.primID, true, std::memory_order_release);
      }
    }
    else {
      m_parts.push_back({LDR_ERROR_INITIALIZATION});
      m_partFoundnames.push_back(foundname);
      if(startLoad) {
        m_startedPartLoad.setBit_ts(entry.partID, true, std::memory_order_release);
      }
      // load will take care of those, so block these actions
      if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
        m_startedPartRenderBuild.setBit_ts(entry.partID, true, std::memory_order_relaxed);
      }
      if(m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
        m_startedPartFix.setBit_ts(entry.partID, true, std::memory_order_relaxed);
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
      if(SUBPART_AS_PRIMITIVE && filename[0] == 's' && (filename[1] == '/' || filename[1] == '\\'))
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

LdrResult Loader::resolvePart(const char* filename, bool allowPrimitives, PartEntry& entry)
{
  bool found = findEntry(filename, entry) == LDR_SUCCESS;

  if(!found) {
    // try register
    std::string foundname;
    bool        isPrimitive;
    if(findLibraryFile(filename, foundname, allowPrimitives, isPrimitive)) {
      LdrResult result = registerInternalPart(filename, foundname, isPrimitive, true, entry);
      if(result == LDR_SUCCESS) {
        // we are the first, let's load the part
        if(isPrimitive) {
          result = loadData(m_primitives[entry.primID], m_renderParts[entry.primID], foundname.c_str(), isPrimitive);
          signalPrimitive(entry.primID);
        }
        else {
          result = loadData(m_parts[entry.partID], m_renderParts[entry.partID], foundname.c_str(), isPrimitive);
          signalPart(entry.partID);
          if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
            signalBuildRender(entry.partID);
          }
          if(m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
            signalFixed(entry.partID);
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
      if(!m_startedPartLoad.setBit_ts(entry.partID, true, std::memory_order_acq_rel)) {
        loadData(m_parts[entry.partID], m_renderParts[entry.partID], m_partFoundnames[entry.partID].c_str(), false);
        signalPart(entry.partID);
        if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
          signalBuildRender(entry.partID);
        }
        if(m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
          signalFixed(entry.partID);
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

  if(m_config.partFixMode != LDR_PART_FIX_NONE) {
    // must have been triggered before, either by onload or manual
    if(!m_startedPartFix.getBit_ts(partid, std::memory_order_acquire)) {
      return LDR_ERROR_DEPENDENT_OPERATION;
    }
    waitFixed(partid);
  }

  bool built = false;
  if(!m_startedPartRenderBuild.setBit_ts(partid, true, std::memory_order_acq_rel)) {
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

LdrResult Loader::registerPart(const char* filename, const LdrPart* part, LdrBool32 isFixed, LdrPartID* pPartID)
{
  PartEntry entry;
  LdrResult result = registerInternalPart(filename, filename, false, true, entry);

  if(result == LDR_SUCCESS) {
    m_parts[entry.partID] = *part;
    if(isFixed) {
      m_startedPartFix.setBit_ts(entry.partID, true, std::memory_order_release);
    }
    signalPart(entry.partID);
    signalFixed(entry.partID);

    *pPartID = entry.partID;
    return LDR_SUCCESS;
  }

  *pPartID = LDR_INVALID_ID;
  return result == LDR_WARNING_IN_USE ? LDR_ERROR_INVALID_OPERATION : result;
}

LdrResult Loader::registerRenderPart(LdrPartID partid, const LdrRenderPart* rpart)
{
  if(partid >= getNumRegisteredParts() || m_startedPartRenderBuild.setBit_ts(partid, true, std::memory_order_release)) {
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


LdrResult Loader::fixParts(uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  if(m_config.partFixMode != LDR_PART_FIX_MANUAL)
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
    if(!m_startedPartFix.setBit_ts(partid, true)) {
      fixPart(partid);
    }
    else {
      inFlight = true;
    }
  }

  if(inFlight) {
    for(uint32_t i = 0; i < numParts; i++) {
      LdrPartID partid = all ? (LdrPartID)i : *(const LdrPartID*)(partsBytes + i * partStride);
      waitFixed(partid);
    }
  }

  return LDR_SUCCESS;
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
    if(!m_startedPartRenderBuild.setBit_ts(partid, true)) {
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


LdrResult Loader::loadDeferredParts(uint32_t numParts, const LdrPartID* parts, size_t partStride, LdrResult* pResults)
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
    if(!m_startedPartLoad.setBit_ts(partID, true, std::memory_order_acq_rel)) {
      LdrResult result = loadData(m_parts[partID], m_renderParts[partID], m_partFoundnames[partID].c_str(), false);
      signalPart(partID);
      if(m_config.renderpartBuildMode == LDR_RENDERPART_BUILD_ONLOAD) {
        signalBuildRender(partID);
      }
      if(m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
        signalFixed(partID);
      }
      if(result != LDR_SUCCESS) {
        resultFinal = result;
      }
      if(pResults && !all) {
        pResults[i] = result;
      }
    }
    else {
      inFlight = true;
    }
  }

  if(inFlight) {
    for(uint32_t i = 0; i < numParts; i++) {
      LdrPartID partid = all ? (LdrPartID)i : *(const LdrPartID*)(((const uint8_t*)parts) + partStride * i);
      waitPart(partid);
      if(pResults && !all) {
        pResults[i] = m_parts[partid].loadResult;
      }
    }
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
#if LDR_CFG_THREADSAFE_RESOLVES
  // ensure all parts are there
  result = loadDeferredParts(model->num_instances, &model->instances[0].part, sizeof(LdrInstance), nullptr);
#endif

  for(uint32_t i = 0; i < model->num_instances; i++) {
    const LdrInstance& instance = model->instances[i];
    const LdrPart&     part     = getPart(instance.part);

    if(part.num_shapes || part.num_positions) {
      builder.instances.push_back(instance);
      bbox_merge(builder.bbox, instance.transform, getPart(instance.part).bbox);
    }
    // flatten
    for(uint32_t s = 0; s < part.num_instances; s++) {
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
  deinitModel(*(LdrModel*)model);

  delete(LdrModel*)model;
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
  deinitRenderModel(*(LdrRenderModel*)renderModel);
  delete(LdrRenderModel*)renderModel;
}

LdrResult Loader::loadData(LdrPart& part, LdrRenderPart& renderPart, const char* filename, bool isPrimitive)
{
  Text txt;
  if(!txt.load(filename)) {
    // actually should not get here, as resolve function takes care of finding files
    assert(0);

    part.loadResult = LDR_ERROR_FILE_NOT_FOUND;
    return part.loadResult;
  }

  BuilderPart builder;
  builder.flag.isPrimitive = isPrimitive ? 1 : 0;
  builder.filename         = filename;

  BFCWinding   winding        = BFC_CCW;
  BFCCertified certified      = BFC_UNKNOWN;
  bool         localCull      = true;
  bool         invertNext     = false;
  bool         keepInvertNext = false;

  char* line = txt.buffer;
  for(size_t i = 0; i < txt.size; i++) {
    if(txt.buffer[i] == '\r')
      txt.buffer[i] = ' ';
    if(txt.buffer[i] != '\n')
      continue;

    txt.buffer[i] = 0;

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

        char subfilename[512] = {0};

        int read = sscanf(line, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %511s", &dummy, &material, mat + 12, mat + 13,
                          mat + 14, mat + 0, mat + 4, mat + 8, mat + 1, mat + 5, mat + 9, mat + 2, mat + 6, mat + 10, subfilename);

        if(read != 15) {
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

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
          LdrResult result = resolvePart(subfilename, true, entry);
          if(result == LDR_SUCCESS) {
            if(entry.isPrimitive() || m_parts[entry.partID].flag.isSubpart) {
              appendBuilderPrimitive(builder, transform, entry.primID, material, invertNext);
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
            sscanf(line, "%d %d %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z, &vecB.x, &vecB.y, &vecB.z);
        if(read != 8) {
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }
        uint32_t vidx = (uint32_t)builder.positions.size();
        builder.lines.push_back(vidx);
        builder.lines.push_back(vidx + 1);
        builder.positions.push_back(vecA);
        builder.positions.push_back(vecB);
        assert(material == LDR_MATERIALID_EDGE);

        builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
      } break;
      case 3: {
        // triangle
        LdrVector vecA;
        LdrVector vecB;
        LdrVector vecC;
        // line
        int read = sscanf(line, "%d %d %f %f %f %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z,
                          &vecB.x, &vecB.y, &vecB.z, &vecC.x, &vecC.y, &vecC.z);
        if(read != 11) {
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }

        float dot = fabsf(vec_dot(vec_normalize(vec_sub(vecB, vecA)), vec_normalize(vec_sub(vecC, vecA))));
        if(dot <= Loader::NO_AREA_TRIANGLE_DOT) {
          uint32_t vidx = (uint32_t)builder.positions.size();
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
          builder.materials.push_back(material);
          builder.quads.push_back(LDR_INVALID_IDX);

          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecB, vecC)));
          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecC, vecA)));
        }
      } break;
      case 4: {
        // quad
        LdrVector vecA;
        LdrVector vecB;
        LdrVector vecC;
        LdrVector vecD;
        // line
        int read = sscanf(line, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y,
                          &vecA.z, &vecB.x, &vecB.y, &vecB.z, &vecC.x, &vecC.y, &vecC.z, &vecD.x, &vecD.y, &vecD.z);
        if(read != 14) {
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }
        float dot = fabsf(vec_dot(vec_normalize(vec_sub(vecB, vecA)), vec_normalize(vec_sub(vecC, vecA))));
        if(dot <= Loader::NO_AREA_TRIANGLE_DOT) {
          uint32_t vidx = (uint32_t)builder.positions.size();
          uint32_t tidx = (uint32_t)builder.triangles.size() / 3;
          // normalize to ccw
          if(winding == BFC_CW) {
            builder.triangles.push_back(vidx + 0);
            builder.triangles.push_back(vidx + 3);
            builder.triangles.push_back(vidx + 2);
            builder.triangles.push_back(vidx + 2);
            builder.triangles.push_back(vidx + 1);
            builder.triangles.push_back(vidx + 0);
          }
          else {
            builder.triangles.push_back(vidx + 0);
            builder.triangles.push_back(vidx + 1);
            builder.triangles.push_back(vidx + 2);
            builder.triangles.push_back(vidx + 2);
            builder.triangles.push_back(vidx + 3);
            builder.triangles.push_back(vidx + 0);
          }
          builder.positions.push_back(vecA);
          builder.positions.push_back(vecB);
          builder.positions.push_back(vecC);
          builder.positions.push_back(vecD);
          builder.materials.push_back(material);
          builder.materials.push_back(material);
          builder.quads.push_back(tidx);
          builder.quads.push_back(tidx);

          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecA, vecB)));
          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecB, vecC)));
          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecC, vecD)));
          builder.minEdgeLength = std::min(builder.minEdgeLength, vec_length(vec_sub(vecD, vecA)));
        }
      } break;
      case 5: {
        // optional line
        LdrVector vecA;
        LdrVector vecB;
        // line
        int read =
            sscanf(line, "%d %d %f %f %f %f %f %f", &dummy, &material, &vecA.x, &vecA.y, &vecA.z, &vecB.x, &vecB.y, &vecB.z);
        if(read != 8) {
          part.loadResult = LDR_ERROR_PARSER;
          return part.loadResult;
        }
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
    MeshUtils::buildRenderPart(builderRender, partTemp, m_config);
    initRenderPart(renderPart, builderRender, partTemp);

    if(!m_config.partFixMode == LDR_PART_FIX_ONLOAD) {
      deinitPart(partTemp);
    }
  }

  return part.loadResult;
}


LdrResult Loader::appendSubModel(BuilderModel& builder, Text& txt, const LdrMatrix& transform, LdrMaterialID material, LdrBool32 autoResolve)
{
  LdrResult finalResult = LDR_SUCCESS;

  char* line = txt.buffer;
  for(size_t i = 0; i < txt.size; i++) {
    if(txt.buffer[i] == '\r')
      txt.buffer[i] = ' ';
    if(txt.buffer[i] != '\n')
      continue;
    txt.buffer[i] = 0;

    // parse line
    int typ = atoi(line);

    // only interested in parts
    if(typ == 1) {
      LdrInstance instance;
      float*      mat = instance.transform.values;
      mat[3] = mat[7] = mat[11] = 0;
      mat[15]                   = 1.0f;

      char subfilename[512] = {0};
      int read = sscanf(line, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %511s", &typ, &instance.material, mat + 12, mat + 13,
                        mat + 14, mat + 0, mat + 4, mat + 8, mat + 1, mat + 5, mat + 9, mat + 2, mat + 6, mat + 10, subfilename);

      instance.transform = mat_mul(transform, instance.transform);
      if(instance.material == LDR_MATERIALID_INHERIT) {
        instance.material = material;
      }

      if(read == 15) {
        // look in subfiles
        bool found = false;
        for(size_t i = 0; i < builder.subFilenames.size(); i++) {
          if(builder.subFilenames[i] == std::string(subfilename)) {
            LdrResult result = appendSubModel(builder, builder.subTexts[i], instance.transform, instance.material, autoResolve);
            if(result == LDR_SUCCESS) {
            }
            else if(result == LDR_WARNING_PART_NOT_FOUND || result == LDR_ERROR_FILE_NOT_FOUND) {
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
          LdrResult result = autoResolve ? resolvePart(subfilename, false, entry) : deferPart(subfilename, false, entry);
          if(result == LDR_SUCCESS) {
            instance.part = entry.partID;
            builder.instances.push_back(instance);

            if(autoResolve) {
              const LdrPart& part = getPart(instance.part);

              if(part.num_shapes || part.num_positions) {
                bbox_merge(builder.bbox, instance.transform, getPart(instance.part).bbox);
              }
              // flatten
              for(uint32_t s = 0; s < part.num_instances; s++) {
                LdrInstance subinstance = part.instances[s];
                if(subinstance.material == LDR_MATERIALID_INHERIT) {
                  subinstance.material = instance.material;
                }
                subinstance.transform = mat_mul(instance.transform, subinstance.transform);
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
        txt.buffer[i] = ' ';
      if(txt.buffer[i] != '\n')
        continue;
      txt.buffer[i] = 0;

      // parse line
      int typ = atoi(line);

      if(typ == 0) {
        char subfilename[512] = {0};
        int  read             = sscanf(line, "0 FILE %511s", subfilename);
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
  LdrResult finalResult = appendSubModel(builder, isMpd ? builder.subTexts[0] : txt, transform, LDR_MATERIAL_COMMON, autoResolve);

  initModel(model, builder);
  return finalResult;
}

LdrResult Loader::makeRenderModel(LdrRenderModel& rmodel, LdrModelHDL model, LdrBool32 autoResolve)
{
  BuilderRenderModel builder;

  builder.bbox = model->bbox;
  for(uint32_t i = 0; i < model->num_instances; i++) {
    const LdrInstance&    instance = model->instances[i];
    BuilderRenderInstance rinstance;
    rinstance.instance = instance;

    if(!autoResolve && !m_startedPartRenderBuild.getBit_ts(instance.part, std::memory_order_relaxed)) {
      // must have been pre-built
      return LDR_ERROR_DEPENDENT_OPERATION;
    }

    if(autoResolve) {
      // try to build on-demand
      LdrResult result = resolveRenderPart(instance.part);
      if(result != LDR_SUCCESS) {
        return result;
      }
    }

    waitBuildRender(instance.part);
    const LdrRenderPart& rpart = m_renderParts[instance.part];
    if(rpart.materials) {
      rinstance.materials.resize(rpart.num_triangles);
      for(uint32_t t = 0; t < rpart.num_triangles; t++) {
        rinstance.materials[t] = rpart.materials[t] == LDR_MATERIALID_INHERIT ? instance.material : rpart.materials[t];
      }
    }
    if(rpart.materialsC) {
      rinstance.materialsC.resize(rpart.num_trianglesC);
      for(uint32_t t = 0; t < rpart.num_trianglesC; t++) {
        rinstance.materialsC[t] = rpart.materialsC[t] == LDR_MATERIALID_INHERIT ? instance.material : rpart.materialsC[t];
      }
    }

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

  static size_t applyRemap3(size_t numIndices, LdrVertexIndex* indices, const uint32_t* LDR_RESTRICT remap)
  {
    size_t outIndices = 0;
    for(size_t i = 0; i < numIndices / 3; i++) {
      uint32_t orig[3];
      orig[0] = indices[i * 3 + 0];
      orig[1] = indices[i * 3 + 1];
      orig[2] = indices[i * 3 + 2];

      uint32_t newIdx[3];
      newIdx[0] = remap[orig[0]];
      newIdx[1] = remap[orig[1]];
      newIdx[2] = remap[orig[2]];

      if(newIdx[0] == newIdx[1] || newIdx[1] == newIdx[2] || newIdx[2] == newIdx[0])
        continue;

      indices[outIndices + 0] = newIdx[0];
      indices[outIndices + 1] = newIdx[1];
      indices[outIndices + 2] = newIdx[2];

      outIndices += 3;
    }

    return outIndices;
  }

  static size_t applyRemap2(size_t numIndices, LdrVertexIndex* indices, const uint32_t* LDR_RESTRICT remap)
  {
    size_t outIndices = 0;
    for(size_t i = 0; i < numIndices / 2; i++) {
      uint32_t orig[2];
      orig[0] = indices[i * 2 + 0];
      orig[1] = indices[i * 2 + 1];

      uint32_t newIdx[2];
      newIdx[0] = remap[orig[0]];
      newIdx[1] = remap[orig[1]];

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

  template <class Tout>
  static uint16_t u16_append(LdrRawData& raw, Tout*& ptrRef, size_t num)
  {
    assert(num <= 0xFFFF);
    ptrRef = (Tout*)raw.size;
    raw.size += sizeof(Tout) * num;
    return uint16_t(num);
  }

  template <class Tout>
  static uint32_t u32_append(LdrRawData& raw, Tout*& ptrRef, size_t num)
  {
    assert(num <= 0xFFFFFFFF);
    ptrRef = (Tout*)raw.size;
    raw.size += sizeof(Tout) * num;
    return uint32_t(num);
  }

  template <class T, class Tout>
  static uint16_t u16_append(LdrRawData& raw, Tout*& ptrRef, const Loader::TVector<T>& vec, size_t divisor = 1)
  {
    assert((vec.size() / divisor) <= 0xFFFF);
    ptrRef = (Tout*)raw.size;
    raw.size += sizeof(Tout) * vec.size();
    return uint16_t(vec.size() / divisor);
  }

  template <class T, class Tout>
  static uint32_t u32_append(LdrRawData& raw, Tout*& ptrRef, const Loader::TVector<T>& vec, size_t divisor = 1)
  {
    assert((vec.size() / divisor) <= 0xFFFFFFFF);
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
  if(part.num_positions || part.num_shapes) {
    LdrInstance instance;
    instance.material  = material;
    instance.part      = partid;
    instance.transform = transform;
    builder.instances.push_back(instance);
  }
  // flatten sub parts
  for(uint32_t s = 0; s < part.num_instances; s++) {
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
  const LdrPart& part = m_primitives[primid];

  builder.flag.hasComplexMaterial |= part.flag.hasComplexMaterial;
  builder.flag.hasNoBackFaceCulling |= part.flag.hasNoBackFaceCulling;

  float scale = std::min(std::min(vec_length(make_vec(transform.col[0])), vec_length(make_vec(transform.col[1]))),
                         vec_length(make_vec(transform.col[2])));

  builder.minEdgeLength = std::min(builder.minEdgeLength, part.minEdgeLength * scale);

  bool transformFlip = mat_determinant(transform) < 0;
  flipWinding        = flipWinding ^ transformFlip;

  uint32_t vecOffset   = (uint32_t)builder.positions.size();
  size_t   shapeOffset = (uint32_t)builder.shapes.size();

  builder.positions.reserve(builder.positions.size() + part.num_positions);

  builder.lines.reserve(builder.lines.size() + part.num_lines * 2);
  builder.triangles.reserve(builder.triangles.size() + part.num_triangles * 3);
  builder.optional_lines.reserve(builder.optional_lines.size() + part.num_optional_lines * 2);
  builder.quads.reserve(builder.quads.size() + part.num_triangles);

  for(uint32_t i = 0; i < part.num_positions; i++) {
    LdrVector vec = transform_point(transform, part.positions[i]);
    builder.positions.push_back(vec);
    bbox_merge(builder.bbox, vec);
  }

  uint32_t triOffset = builder.triangles.size() / 3;
  for(uint32_t i = 0; i < part.num_triangles; i++) {
    builder.quads.push_back(part.quads[i] != LDR_INVALID_IDX ? (part.quads[i] + triOffset) : LDR_INVALID_IDX);
  }

  if(flipWinding) {
    Utils::pushWithOffsetRev3(builder.triangles, vecOffset, part.num_triangles, part.triangles);
  }
  else {
    Utils::pushWithOffset(builder.triangles, vecOffset, part.num_triangles * 3, part.triangles);
  }
  Utils::pushWithOffset(builder.lines, vecOffset, part.num_lines * 2, part.lines);
  Utils::pushWithOffset(builder.optional_lines, vecOffset, part.num_optional_lines * 2, part.optional_lines);

  for(uint32_t i = 0; i < part.num_triangles; i++) {
    builder.materials.push_back(part.materials[i] == LDR_MATERIALID_INHERIT ? material : part.materials[i]);
  }

  // shapes
  Utils::copyAppend(builder.shapes, part.num_shapes, part.shapes);
  for(size_t i = shapeOffset; i < builder.shapes.size(); i++) {
    builder.shapes[i].bfcInvert ^= flipWinding ? 1 : 0;
    builder.shapes[i].transform = mat_mul(transform, builder.shapes[i].transform);
  }

  // flatten sub parts
  for(uint32_t s = 0; s < part.num_instances; s++) {
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

  uint32_t* remapMerge   = new uint32_t[builder.positions.size()];
  uint32_t* remapCompact = new uint32_t[builder.positions.size()];

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

  // 1 LDU ~ 0.4mm
  // merge using minimal edge length
  const float epsilon = builder.minEdgeLength * 0.9;

  // line sweep merge
  size_t     mergeBegin = 1;
  SortVertex refVertex  = sortedVertices[0];

  for(size_t i = 1; i < numVertices; i++) {
    const SortVertex& svertex = sortedVertices[i];
    while(mergeBegin <= i && ((svertex.dot - refVertex.dot > epsilon) || (i == numVertices - 1))) {
      mergeBegin = runMerge(refVertex, sortedVertices.data(), mergeBegin, i, remapMerge, epsilon);
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
  printf("ldraw compaction: %d\n", uint32_t(numVertices - numVerticesNew));
#if 0
  float minDist = FLT_MAX;

  for(size_t i = 0; i < numVerticesNew; i++)
  {
    for(size_t v = i + 1; v < numVerticesNew; v++)
    {
      float sqlen = vec_sq_length( vec_sub(builder.positions[i] , builder.positions[v]));
      if (sqlen <= epsilon * epsilon){
        i = i;
      }
      minDist = std::min(minDist, sqlen);
    }
  }
  minDist = sqrtf(minDist);
#endif
#endif

  size_t lineSize     = Utils::applyRemap2(builder.lines.size(), builder.lines.data(), remapMerge);
  size_t optionalSize = Utils::applyRemap2(builder.optional_lines.size(), builder.optional_lines.data(), remapMerge);
  size_t triangleSize = Utils::applyRemap3(builder.triangles.size(), builder.triangles.data(), remapMerge);

  builder.lines.resize(lineSize);
  builder.optional_lines.resize(optionalSize);
  builder.triangles.resize(triangleSize);
}


void Loader::fillBuilderPart(BuilderPart& builder, LdrPartID partid)
{
  LdrPart& part = m_parts[partid];

  builder.filename      = std::string(part.name);
  builder.flag          = part.flag;
  builder.bbox          = part.bbox;
  builder.minEdgeLength = part.minEdgeLength;
  Utils::fillVector(builder.positions, part.positions, part.num_positions);
  Utils::fillVector(builder.connections, part.connections, part.num_positions);
  Utils::fillVector(builder.lines, part.lines, part.num_lines * 2);
  Utils::fillVector(builder.optional_lines, part.optional_lines, part.num_optional_lines * 2);
  Utils::fillVector(builder.triangles, part.triangles, part.num_triangles * 3);
  Utils::fillVector(builder.materials, part.materials, part.num_triangles);
  Utils::fillVector(builder.quads, part.quads, part.num_triangles);
  Utils::fillVector(builder.instances, part.instances, part.num_instances);
  Utils::fillVector(builder.shapes, part.shapes, part.num_shapes);
}

void Loader::fixPart(LdrPartID partid)
{
  LdrPart& part = m_parts[partid];

  if(!(part.flag.isPrimitive || part.flag.isSubpart)) {
    BuilderPart builder;
    fillBuilderPart(builder, partid);

    deinitPart(part);
    memset(&part, 0, sizeof(LdrPart));

    MeshUtils::fixBuilderPart(builder, m_config);

    initPart(part, builder);
  }

  signalFixed(partid);
}

void Loader::buildRenderPart(LdrPartID partid)
{
  LdrRenderPart& renderPart = m_renderParts[partid];

  if(m_parts[partid].flag.isPrimitive || m_parts[partid].flag.isSubpart) {
    renderPart = LdrRenderPart();
  }
  else {
    BuilderRenderPart builder;
    if(m_config.partFixMode == LDR_PART_FIX_NONE) {
      LdrPart parttemp;

      BuilderPart builderPart;
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

  part.num_positions      = Utils::u32_append(part.raw, part.positions, builder.positions);
  part.num_lines          = Utils::u32_append(part.raw, part.lines, builder.lines, 2);
  part.num_optional_lines = Utils::u32_append(part.raw, part.optional_lines, builder.optional_lines, 2);
  part.num_triangles      = Utils::u32_append(part.raw, part.triangles, builder.triangles, 3);
  Utils::u32_append(part.raw, part.quads, builder.quads);
  Utils::u32_append(part.raw, part.materials, builder.materials);
  Utils::u32_append(part.raw, part.connections, builder.connections);
  part.num_shapes    = Utils::u32_append(part.raw, part.shapes, builder.shapes);
  part.num_instances = Utils::u32_append(part.raw, part.instances, builder.instances);
  part.num_name      = Utils::u32_append(part.raw, part.name, builder.filename.size() + 1);

  rawAllocate(part.raw.size, &part.raw);

  Utils::setup_pointer(part.raw, part.positions, builder.positions);
  Utils::setup_pointer(part.raw, part.lines, builder.lines);
  Utils::setup_pointer(part.raw, part.optional_lines, builder.optional_lines);
  Utils::setup_pointer(part.raw, part.triangles, builder.triangles);
  Utils::setup_pointer(part.raw, part.quads, builder.quads);
  Utils::setup_pointer(part.raw, part.materials, builder.materials);
  Utils::setup_pointer(part.raw, part.connections, builder.connections);
  Utils::setup_pointer(part.raw, part.shapes, builder.shapes);
  Utils::setup_pointer(part.raw, part.instances, builder.instances);
  Utils::setup_pointer(part.raw, part.name, builder.filename.size() + 1, builder.filename.c_str());

#if _DEBUG && EXTRA_ASSERTS
  assert(builder.positions.size() == part.num_positions);
  assert(builder.lines.size() == part.num_lines * 2);
  for(size_t i = 0; i < builder.lines.size(); i++) {
    assert(part.lines[i] <= builder.positions.size());
  }
  assert(builder.optional_lines.size() == part.num_optional_lines * 2);
  for(size_t i = 0; i < builder.optional_lines.size(); i++) {
    assert(part.optional_lines[i] <= builder.positions.size());
  }
  assert(builder.triangles.size() == part.num_triangles * 3);
  for(size_t i = 0; i < builder.triangles.size(); i++) {
    assert(part.triangles[i] <= builder.positions.size());
  }
#endif
}

void Loader::initModel(LdrModel& model, const BuilderModel& builder)
{
  model.bbox          = builder.bbox;
  model.raw.size      = 0;
  model.num_instances = Utils::u32_append(model.raw, model.instances, builder.instances);

  rawAllocate(model.raw.size, &model.raw);

  Utils::setup_pointer(model.raw, model.instances, builder.instances);
}

void Loader::initRenderPart(LdrRenderPart& renderpart, const BuilderRenderPart& builder, const LdrPart& part)
{
  uint32_t hasMaterials = part.flag.hasComplexMaterial ? 1 : 0;
  renderpart.flag       = builder.flag;
  renderpart.bbox       = builder.bbox;
  renderpart.raw.size   = 0;

  renderpart.num_vertices   = Utils::u32_append(renderpart.raw, renderpart.vertices, builder.vertices);
  renderpart.num_lines      = Utils::u32_append(renderpart.raw, renderpart.lines, builder.lines, 2);
  renderpart.num_triangles  = Utils::u32_append(renderpart.raw, renderpart.triangles, builder.triangles, 3);
  renderpart.num_trianglesC = Utils::u32_append(renderpart.raw, renderpart.trianglesC, builder.trianglesC, 3);
  Utils::u32_append(renderpart.raw, renderpart.materials, part.num_triangles * hasMaterials);
  Utils::u32_append(renderpart.raw, renderpart.materialsC, builder.materialsC);
  renderpart.num_shapes = Utils::u32_append(renderpart.raw, renderpart.shapes, part.num_shapes);

  rawAllocate(renderpart.raw.size, &renderpart.raw);

  Utils::setup_pointer(renderpart.raw, renderpart.vertices, builder.vertices);
  Utils::setup_pointer(renderpart.raw, renderpart.lines, builder.lines);
  Utils::setup_pointer(renderpart.raw, renderpart.triangles, builder.triangles);
  Utils::setup_pointer(renderpart.raw, renderpart.trianglesC, builder.trianglesC);
  Utils::setup_pointer(renderpart.raw, renderpart.materials, part.num_triangles * hasMaterials, part.materials);
  Utils::setup_pointer(renderpart.raw, renderpart.materialsC, builder.materialsC);
  Utils::setup_pointer(renderpart.raw, renderpart.shapes, part.num_shapes, part.shapes);
}

void Loader::initRenderModel(LdrRenderModel& rmodel, const BuilderRenderModel& builder)
{
  rmodel.raw.size      = 0;
  rmodel.num_instances = Utils::u32_append(rmodel.raw, rmodel.instances, builder.instances.size());
  rmodel.bbox          = builder.bbox;

  size_t begin = rmodel.raw.size;

  for(uint32_t i = 0; i < rmodel.num_instances; i++) {
    const LdrRenderPart& part         = getRenderPart(builder.instances[i].instance.part);
    uint32_t             hasMaterials = part.flag.hasComplexMaterial ? 1 : 0;

    rmodel.raw.size += part.num_triangles * hasMaterials * sizeof(LdrMaterialID);
    rmodel.raw.size += part.num_trianglesC * hasMaterials * sizeof(LdrMaterialID);
  }

  rawAllocate(rmodel.raw.size, &rmodel.raw);

  Utils::setup_pointer(rmodel.raw, rmodel.instances, 0, true);

  for(uint32_t i = 0; i < rmodel.num_instances; i++) {
    const LdrRenderPart& part      = getRenderPart(builder.instances[i].instance.part);
    LdrRenderInstance&   rinstance = rmodel.instances[i];
    rinstance.instance             = builder.instances[i].instance;

    uint32_t hasMaterials = part.flag.hasComplexMaterial ? 1 : 0;

    Utils::setup_pointer(rmodel.raw, rinstance.materials, begin, hasMaterials);
    if(hasMaterials) {
      for(uint32_t t = 0; t < part.num_triangles; t++) {
        LdrMaterialID mtl      = part.materials[t];
        rinstance.materials[t] = mtl == LDR_MATERIALID_INHERIT ? rinstance.instance.material : mtl;
      }
      begin += part.num_triangles * sizeof(LdrMaterialID);
    }

    Utils::setup_pointer(rmodel.raw, rinstance.materialsC, begin, hasMaterials);
    if(hasMaterials && part.num_trianglesC) {
      for(uint32_t t = 0; t < part.num_trianglesC; t++) {
        LdrMaterialID mtl       = part.materialsC[t];
        rinstance.materialsC[t] = mtl == LDR_MATERIALID_INHERIT ? rinstance.instance.material : mtl;
      }
      begin += part.num_trianglesC * sizeof(LdrMaterialID);
    }
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

}  // namespace ldr

//////////////////////////////////////////////////////////////////////////
// C-Api

#if LDR_CFG_C_API

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
  ldr::Loader* lldr = (ldr::Loader*)loader;
  lldr->deinit();
  delete lldr;
}

LDR_API LdrResult ldrRegisterShapeType(LdrLoaderHDL loader, const char* filename, LdrShapeType type)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerShapeType(filename, type);
}

LDR_API LdrResult ldrRegisterPart(LdrLoaderHDL loader, const char* filename, const LdrPart* part, LdrBool32 isFixed, LdrPartID* pPartID)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->registerPart(filename, part, isFixed, pPartID);
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

LDR_API LdrResult ldrFixParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->fixParts(numParts, parts, partStride);
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

LDR_API LdrResult ldrLoadDeferredParts(LdrLoaderHDL loader, uint32_t numParts, const LdrPartID* parts, size_t partStride, LdrResult* pResults)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return lldr->loadDeferredParts(numParts, parts, partStride, pResults);
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

LDR_API const LdrMaterial* ldrGetMaterial(LdrLoaderHDL loader, LdrMaterialID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return &lldr->getMaterial(idx);
}
LDR_API const LdrPart* ldrGetPart(LdrLoaderHDL loader, LdrPartID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return &lldr->getPart(idx);
}
LDR_API const LdrPart* ldrGetPrimitive(LdrLoaderHDL loader, LdrPrimitiveID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return &lldr->getPrimitive(idx);
}
LDR_API const LdrRenderPart* ldrGetRenderPart(LdrLoaderHDL loader, LdrPartID idx)
{
  ldr::Loader* lldr = (ldr::Loader*)loader;
  return &lldr->getRenderPart(idx);
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
1 10 0 320 0 1 0 0 0 1 0 0 0 1 92908.dat
*/
