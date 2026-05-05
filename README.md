# Bijective Projection in a Shell


Zhongshi Jiang, Teseo Schneider, Denis Zorin, Daniele Panozzo. 
*ACM Transactions on Graphics (In Proceedings of SIGGRAPH Asia 2020)*

<img src="https://i.imgur.com/sgiVMlh.jpg" width="200"/>

## WASM port + browser demo

This fork ports the full library to Emscripten/WASM and ships an interactive
three.js demo. CGAL, geogram, embree, HDF5, and TBB have been replaced with
Shewchuk's robust predicates, libigl AABB, and the in-source spatial-hash
self-intersection module. The native build is unaffected (`PRISM_WASM` is
opt-in).

- **Live demo:** [GitHub Pages](https://github.com/) (auto-deployed from
  `main`)
- **Build it locally:**
  ```
  source <emsdk>/emsdk_env.sh
  mkdir build_wasm && cd build_wasm
  emcmake cmake -DCMAKE_BUILD_TYPE=Release -DPRISM_WASM=ON -DPRISM_TESTS=OFF ..
  emmake make prism_full_wasm -j$(nproc)
  cp prism_full_wasm.{js,wasm} ../web/
  python3 -m http.server -d ../web 8000   # then open http://127.0.0.1:8000/
  ```

The demo exercises real `PrismCage::PrismCage(V, F, doosabineps, initial_step)`
construction in the browser, then runs the bijective Phong projection on
query points sampled inside the prismatic shell. It supports both built-in
three.js geometries (torus knots, lathes, capsules…) and CSG manifolds built
on the fly via [`manifold-3d`](https://github.com/elalish/manifold).

## Abstract
We introduce an algorithm to convert a self-intersection free, orientable, and manifold triangle mesh T into a generalized prismatic shell equipped with a bijective projection operator to map T to a class of discrete surfaces contained within the shell whose normals satisfy a simple local condition. Properties can be robustly and efficiently transferred between these surfaces using the prismatic layer as a common parametrization domain.

The combination of the prismatic shell construction and corresponding projection operator is a robust building block readily usable in many downstream applications, including the solution of PDEs, displacement maps synthesis, Boolean operations, tetrahedral meshing, geometric textures, and nested cages.

## Installation (native)
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

The WASM build is described above under *WASM port + browser demo*.

## Usage

Some scripts are provided in [FigureScripts](FigureScripts.md).
Please stay tuned for more information!

