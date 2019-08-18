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

#pragma once

#include "ldrawloader.h"

#include <assert.h>
#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace ldr {

struct Loader
{
  friend class MeshUtils;

  template <class TvtxIndex_t, class TvtxIndexPair, int VTX_BITS, int VTX_TRIS>
  friend class TMesh;

  friend class Utils;

public:
  Loader() {}
  Loader(const LdrLoaderCreateInfo* config);

  ~Loader() { deinit(); }

  // api described in C header

  LdrResult init(const LdrLoaderCreateInfo* config);
  void      deinit();

  LdrResult registerShapeType(const char* filename, LdrShapeType type);
  LdrResult registerPrimitive(const char* filename, const LdrPart* part);
  LdrResult registerPart(const char* filename, const LdrPart* part, LdrBool32 isFixed, LdrPartID* pPartID);
  LdrResult registerRenderPart(LdrPartID partid, const LdrRenderPart* rpart);

  LdrResult rawAllocate(size_t size, LdrRawData* raw);
  LdrResult rawFree(const LdrRawData* raw);

  LdrResult fixParts(uint32_t numParts, const LdrPartID* parts, size_t partStride);
  LdrResult buildRenderParts(uint32_t numParts, const LdrPartID* parts, size_t partStride);

  LdrResult createModel(const char* filename, LdrBool32 autoResolve, LdrModelHDL* pModel);
  // only required if autoResolve was false
  LdrResult resolveModel(LdrModelHDL model);
  void      destroyModel(LdrModelHDL model);

  LdrResult createRenderModel(LdrModelHDL model, LdrBool32 autoResolve, LdrRenderModelHDL* pRenderModel);
  void      destroyRenderModel(LdrRenderModelHDL renderModel);

  LdrResult loadDeferredParts(uint32_t numParts, const LdrPartID* parts, size_t partStride, LdrResult* pResults);

  inline const LdrMaterial&   getMaterial(LdrMaterialID idx) const { return m_materials[idx]; }
  inline const LdrPart&       getPart(LdrPartID idx) const { return m_parts[idx]; }
  inline const LdrPart&       getPrimitive(LdrPrimitiveID idx) const { return m_primitives[idx]; }
  inline const LdrRenderPart& getRenderPart(LdrPartID idx) const { return m_renderParts[idx]; }

  inline uint32_t getNumRegisteredParts() const
  {
    SpinMutex::Scoped scopedLock(m_partRegistryMutex);
    return (uint32_t)m_parts.size();
  }

  inline LdrPartID findPart(const char* filename) const
  {
    LdrPartID id = LDR_INVALID_ID;
    PartEntry entry;
    if(findEntry(filename, entry) == LDR_SUCCESS) {
      id = entry.partID;
    }

    return id;
  }

  inline LdrPartID findPrimitive(const char* filename) const
  {
    LdrPartID id = LDR_INVALID_ID;
    PartEntry entry;
    if(findEntry(filename, entry) == LDR_SUCCESS) {
      id = entry.primID;
    }

    return id;
  }

  inline LdrShapeType findShapeType(const char* filename) const
  {
    const auto it = m_shapeRegistry.find(filename);
    if(it != m_shapeRegistry.cend()) {
      return it->second;
    }
    else {
      return LDR_INVALID_ID;
    }
  }


private:
  // subpart are parts from parts/s directory
  static const bool SUBPART_AS_PRIMITIVE = true;

  static const float NO_AREA_TRIANGLE_DOT;
  static const float FORCED_HARD_EDGE_DOT;
  static const float CHAMFER_PARALLEL_DOT;
  static const float ANGLE_45_DOT;

  static const uint32_t MAX_PARTS = 16384;
  static const uint32_t MAX_PRIMS = 8192;

  // first two search paths are within primitive directory
  static const uint32_t PRIMITIVE_PATHS = 2;
  // total number of search paths
  static const uint32_t SEARCH_PATHS = 4;

  struct SpinMutex
  {
    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;

    void lock()
    {
      while(m_lock.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    void unlock() { m_lock.clear(std::memory_order_release); }

    class Scoped
    {
    public:
      SpinMutex& m_mutex;
      Scoped(SpinMutex& mutex)
          : m_mutex(mutex)
      {
        mutex.lock();
      }
      ~Scoped() { m_mutex.unlock(); }
    };
  };

#if 0
  template <typename T>
  using TVector = std::vector<T>;
#else
  template <class T>
  class TVector
  {
    // works only for PODs or data types
    // that are valid when cleared to zero
    // faster debugging perf

  private:
    T*       m_data     = nullptr;
    uint32_t m_capacity = 0;
    uint32_t m_size     = 0;

    void copy(const TVector<T>& other)
    {
      if (!other.m_size) return;

      reserve(other.m_size);
      memcpy(m_data, other.m_data, sizeof(T) * other.m_size);
      m_size = other.m_size;
    }

  public:
    TVector() {}
    TVector(uint32_t num, T ref = T()) { resize(num, ref); }
    ~TVector() { reset(); }
    TVector(const TVector<T>& other) { copy(other); }

    TVector<T>& operator=(const TVector<T>& other)
    {
      reset();
      copy(other);
      return *this;
    }

    T& operator[](uint32_t i)
    {
      assert(i < m_size);
      return m_data[i];
    }

    const T& operator[](uint32_t i) const
    {
      assert(i < m_size);
      return m_data[i];
    }

    bool empty() const { return m_size == 0; }

    uint32_t size() const { return m_size; }

    T*       data() { return m_data; }
    const T* data() const { return m_data; }

    void reserve(uint32_t size)
    {
      if(size < m_capacity)
        return;

      uint32_t sz = (size + 7) & ~7;
      m_data      = (T*)realloc(m_data, sizeof(T) * sz);
      m_capacity  = sz;
    }
    void resize(uint32_t size, T ref = T())
    {
      reserve(size);
      for(uint32_t i = m_size; i < size; i++) {
        m_data[i] = ref;
      }
      m_size = size;
    }
    void clear(){ m_size = 0; };
    void reset()
    {
      if(m_data) {
        free(m_data);
        m_data = nullptr;
        m_capacity = 0;
        m_size     = 0;
      }
    };
    void push_back(const T& val)
    {
      if(m_size + 1 > m_capacity) {
        uint32_t nextSize = (m_size * 3) / 2;

        reserve(nextSize < m_size + 1 ? m_size + 1 : nextSize);
      }
      m_data[m_size++] = val;
    }
    void pop_back()
    {
      assert(m_size);
      m_size--;
    }
    const T& back() const
    {
      assert(m_size);
      return m_data[m_size - 1];
    }
  };
#endif

  struct BitArray
  {
    uint64_t* data          = nullptr;
    uint32_t  num           = 0;
    uint32_t  num_allocated = 0;

    BitArray() {}
    BitArray(uint32_t num, bool value) { resize(num, value); }
    void clear() { memset(data, 0, sizeof(uint64_t) * (num_allocated / 64)); }

    uint32_t size() const { return num; }

    void resize(uint32_t numNew, bool value)
    {
      num = numNew;
      if(num_allocated < numNew) {
        uint32_t num_old = num_allocated;
        numNew           = (numNew + 63) & ~63;
        num_allocated    = numNew;
        data             = (uint64_t*)realloc(data, sizeof(uint64_t) * (num_allocated / 64));

        uint32_t idx       = num_old / 64;
        uint32_t num_delta = numNew - num_old;
        memset(data + idx, value ? 0xFFFFFFFF : 0, sizeof(uint64_t) * (num_delta / 64));
      }
    }

    bool getBit(uint32_t idx) const
    {
      assert(idx <= num);
      uint64_t mask = (uint64_t(1) << (idx % 64));
      uint64_t old  = data[idx / 64];
      return (old & mask) != 0;
    }

    bool setBit(uint32_t idx, bool state)
    {
      assert(idx <= num);
      uint64_t mask = (uint64_t(1) << (idx % 64));
      uint64_t old  = data[idx / 64];
      if(state) {
        data[idx / 64] |= mask;
      }
      else {
        data[idx / 64] &= ~mask;
      }
      return (old & mask) != 0;
    }

    bool getBit_ts(uint32_t idx, std::memory_order memorder = std::memory_order_relaxed) const
    {
      assert(idx <= num);
      const std::atomic_uint64_t* atomic_data = (const std::atomic_uint64_t*)data;
      uint64_t                    mask        = (uint64_t(1) << (idx % 64));
      uint64_t                    old         = atomic_data[idx / 64].load(memorder);
      return (old & mask) != 0;
    }

    bool setBit_ts(uint32_t idx, bool state, std::memory_order memorder = std::memory_order_relaxed)
    {
      assert(idx <= num);
      std::atomic_uint64_t* atomic_data = (std::atomic_uint64_t*)data;
      uint64_t              mask        = (uint64_t(1) << (idx % 64));
      uint64_t              old;
      if(state) {
        old = atomic_data[idx / 64].fetch_or(mask, memorder);
      }
      else {
        old = atomic_data[idx / 64].fetch_and(~mask, memorder);
      }
      return (old & mask) != 0;
    }
  };

  enum BFCWinding
  {
    BFC_CCW,
    BFC_CW,
  };
  enum BFCCertified
  {
    BFC_TRUE,
    BFC_FALSE,
    BFC_UNKNOWN,
  };

  struct Config : public LdrLoaderCreateInfo
  {
    std::string basePathString;
    std::string cachFileString;
  };

  struct Text
  {
    char*  buffer = nullptr;
    size_t size   = 0;
    bool   referenced  = false;

    ~Text()
    {
      if(buffer && !referenced) {
        free(buffer);
      }
    }

    bool load(const char* filename){
      FILE* file = fopen(filename, "rb");

      if(!file)
        return false;

      fseek(file, 0, SEEK_END);
      size = ftell(file);
      rewind(file);

      buffer = (char*)malloc(sizeof(char) * (size + 2));
      fread(buffer, size, 1, file);
      fclose(file);

      buffer[size] = '\n';
      buffer[size+1] = 0;

      size += 2;

      return true;
    }
  };

  struct BuilderPart
  {
    std::string filename;
    LdrPartFlag flag          = {0, 0};
    LdrBbox     bbox          = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
    float       minEdgeLength = FLT_MAX;

    TVector<LdrVector>     positions;
    TVector<uint32_t>      lines;
    TVector<uint32_t>      optional_lines;
    TVector<uint32_t>      triangles;
    TVector<uint32_t>      connections;
    TVector<uint32_t>      quads;
    TVector<LdrMaterialID> materials;
    TVector<LdrShape>      shapes;
    TVector<LdrInstance>   instances;

    inline uint32_t addConnection(uint32_t v)
    {
      if(connections[v] == LDR_INVALID_IDX) {
        connections[v] = (uint32_t)positions.size();
        positions.push_back(positions[v]);
        connections.push_back(v);
      }
      return connections[v];
    }

    inline bool hasConnection(uint32_t v) const { return connections[v] != LDR_INVALID_IDX; }

    inline bool isQuad(uint32_t t) const { return quads[t] != LDR_INVALID_IDX; }
  };

  struct BuilderModel
  {
    LdrBbox              bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
    TVector<LdrInstance> instances;

    std::vector<Text>         subTexts;
    std::vector<std::string>  subFilenames;
  };

  struct BuilderRenderPart
  {
    struct EdgePair
    {
      uint32_t edgeA;
      uint32_t edgeB;
      uint32_t triA;
      uint32_t triB;
    };

    LdrPartFlag flag;
    LdrBbox     bbox;

    TVector<LdrRenderVertex> vertices;
    TVector<uint32_t>        lines;
    TVector<uint32_t>        triangles;
    TVector<uint32_t>        trianglesC;
    TVector<LdrMaterialID>   materialsC;

    TVector<LdrVector> triNormals;
    TVector<uint32_t>  vtxOutCount;
    TVector<uint32_t>  vtxOutBegin;

    TVector<EdgePair> vtxOutEdgePairs;
  };

  struct BuilderRenderInstance
  {
    LdrInstance            instance;
    TVector<LdrMaterialID> materials;
    TVector<LdrMaterialID> materialsC;
  };

  struct BuilderRenderModel
  {
    LdrBbox                             bbox;
    // builder instance is not POD hence don't use TVector
    std::vector<BuilderRenderInstance>  instances;
  };


  struct PartEntry
  {
    LdrPartID      partID = LDR_INVALID_ID;
    LdrPrimitiveID primID = LDR_INVALID_ID;

    bool isPrimitive() const { return primID != LDR_INVALID_ID; }
  };

  Config                   m_config;
  std::string              m_searchPaths[SEARCH_PATHS];
  TVector<LdrMaterial>     m_materials;
  TVector<LdrPart>         m_primitives;
  TVector<LdrPart>         m_parts;
  TVector<LdrRenderPart>   m_renderParts;
  std::vector<std::string> m_partFoundnames;
  std::vector<std::string> m_primitiveFoundnames;

  // flags if operations have started (completion may be manually managed or via resolves)
  // m_finished bitarrays only exist in thread-safe variant
  BitArray m_startedPartLoad;
  BitArray m_startedPrimitiveLoad;
  BitArray m_startedPartFix;
  BitArray m_startedPartRenderBuild;

  mutable SpinMutex m_partRegistryMutex;

  std::unordered_map<std::string, PartEntry>    m_partRegistry;
  std::unordered_map<std::string, LdrShapeType> m_shapeRegistry;


  LdrResult registerInternalPart(const char* filename, const std::string& foundname, bool isPrimitive, bool startLoad, PartEntry& entry);

  LdrResult deferPart(const char* filename, bool allowPrimitive, PartEntry& entry);
  LdrResult resolvePart(const char* filename, bool allowPrimitive, PartEntry& entry);
  LdrResult resolveRenderPart(LdrPartID partid);

  bool findLibraryFile(const char* filename, std::string& foundname, bool allowPrimitives, bool& isPrimitive);

  LdrResult loadData(LdrPart& part, LdrRenderPart& renderpart, const char* filename, bool isPrimitive);
  LdrResult loadModel(LdrModel& model, const char* filename, LdrBool32 autoResolve);
  LdrResult makeRenderModel(LdrRenderModel& rmodel, LdrModelHDL model, LdrBool32 autoResolve);

  LdrResult appendSubModel(BuilderModel& builder, Text& text, const LdrMatrix& transform, LdrMaterialID material, LdrBool32 autoResolve);

  void appendBuilderPrimitive(BuilderPart& builder, const LdrMatrix& transform, LdrPrimitiveID primid, LdrMaterialID material, bool flipWinding);
  void appendBuilderPart(BuilderPart& builder, const LdrMatrix& transform, LdrPartID partid, LdrMaterialID material, bool flipWinding);
  void compactBuilderPart(BuilderPart& builder);

  void fillBuilderPart(BuilderPart& builder, LdrPartID partid);

  void fixPart(LdrPartID partid);
  void buildRenderPart(LdrPartID partid);

  void initPart(LdrPart& part, const BuilderPart& builder);
  void initModel(LdrModel& model, const BuilderModel& builder);
  void initRenderPart(LdrRenderPart& rmesh, const BuilderRenderPart& builder, const LdrPart& part);
  void initRenderModel(LdrRenderModel& rmodel, const BuilderRenderModel& builder);

  void deinitPart(LdrPart& part);
  void deinitModel(LdrModel& model);
  void deinitRenderPart(LdrRenderPart& rmesh);
  void deinitRenderModel(LdrRenderModel& rmodel);

  inline LdrResult findEntry(const char* filename, PartEntry& entry) const
  {
    LdrResult result = LDR_ERROR_OTHER;

    SpinMutex::Scoped mutex(m_partRegistryMutex);
    const auto        it = m_partRegistry.find(filename);
    if(it != m_partRegistry.cend()) {
      entry  = it->second;
      result = LDR_SUCCESS;
    }

    return result;
  }

#if LDR_CFG_THREADSAFE_RESOLVES

  // flags that tell us if async operations completed
  BitArray m_finishedPartLoad;
  BitArray m_finishedPrimitiveLoad;
  BitArray m_finishedPartFix;
  BitArray m_finishedPartRenderBuild;

  // FIXME not optimal to use busy hot loops here
  // maybe expose as external virtual/function pointers

  inline void signalPart(LdrPartID partid) { m_finishedPartLoad.setBit_ts(partid, true, std::memory_order_release); }

  inline void signalPrimitive(LdrPrimitiveID primid)
  {
    m_finishedPrimitiveLoad.setBit_ts(primid, true, std::memory_order_release);
  }

  inline void signalFixed(LdrPartID partid) { m_finishedPartFix.setBit_ts(partid, true, std::memory_order_release); }

  inline void signalBuildRender(LdrPartID partid)
  {
    m_finishedPartRenderBuild.setBit_ts(partid, true, std::memory_order_release);
  }

  inline void waitFixed(LdrPartID partid) const
  {
    while(!m_finishedPartFix.getBit_ts(partid, std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  inline void waitBuildRender(LdrPartID partid) const
  {
    while(!m_finishedPartRenderBuild.getBit_ts(partid, std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  inline void waitPart(LdrPartID partid) const
  {
    while(!m_finishedPartLoad.getBit_ts(partid, std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  inline void waitPrimitive(LdrPrimitiveID primid) const
  {
    while(!m_finishedPrimitiveLoad.getBit_ts(primid, std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
#else
  inline void signalPart(LdrPartID partid) {}

  inline void signalPrimitive(LdrPartID partid) {}

  inline void signalFixed(LdrPartID partid) {}

  inline void signalBuildRender(LdrPartID partid) {}

  inline void waitFixed(LdrPartID partid) const {}

  inline void waitBuildRender(LdrPartID partid) const {}

  inline void waitPart(LdrPartID partid) const {}

  inline void waitPrimitive(LdrPartID partid) const {}
#endif
};


}  // namespace ldr
