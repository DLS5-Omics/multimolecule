(function () {
  const SCRIPT_URL =
    (document.currentScript && document.currentScript.src) ||
    new URL("assets/javascripts/p53-hero.js", document.baseURI).toString()
  const MOL_SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js",
    "https://unpkg.com/3dmol@2.5.5/build/3Dmol-min.js",
  ]
  let molPromise

  function get3Dmol() {
    return window.$3Dmol || window["3Dmol"]
  }

  function load3Dmol() {
    const loaded = get3Dmol()
    if (loaded && loaded.createViewer) return Promise.resolve(loaded)
    if (molPromise) return molPromise

    molPromise = MOL_SCRIPT_URLS.reduce(
      (chain, url) => chain.catch(() => load3DmolScript(url)),
      Promise.reject(),
    )

    return molPromise
  }

  function load3DmolScript(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script")
      script.src = url
      script.async = true
      script.onload = () => {
        const mol = get3Dmol()
        if (mol && mol.createViewer) {
          resolve(mol)
        } else {
          reject(new Error("3Dmol did not expose createViewer"))
        }
      }
      script.onerror = () => reject(new Error("Could not load 3Dmol"))
      document.head.appendChild(script)
    })
  }

  function canRenderWebGL() {
    return "WebGLRenderingContext" in window
  }

  function shouldSpin() {
    return !window.matchMedia("(prefers-reduced-motion: reduce)").matches
  }

  function structureUrl(root) {
    if (root.dataset.structureUrl) return root.dataset.structureUrl
    return new URL("../structures/p53-dna-1tup.pdb", SCRIPT_URL).toString()
  }

  function styleModel(viewer) {
    viewer.setStyle({}, {})

    for (const chain of ["A", "B", "C"]) {
      viewer.setStyle(
        { chain },
        {
          cartoon: {
            color: 0x9eacb8,
            opacity: 0.72,
          },
        },
      )
    }

    viewer.setStyle(
      { chain: "E" },
      {
        cartoon: {
          color: 0x178f8a,
          opacity: 0.9,
        },
        stick: {
          color: 0x2aa7a0,
          opacity: 0.72,
          radius: 0.075,
        },
      },
    )
    viewer.setStyle(
      { chain: "F" },
      {
        cartoon: {
          color: 0xab4f50,
          opacity: 0.9,
        },
        stick: {
          color: 0xbf6664,
          opacity: 0.72,
          radius: 0.075,
        },
      },
    )
  }

  function initP53Hero() {
    const root = document.querySelector("[data-mm-p53-viewer]")
    if (!root || root.dataset.mmMolStatus) return
    if (!canRenderWebGL()) return

    root.dataset.mmMolStatus = "loading"
    const hero = root.closest(".mm-hero")

    load3Dmol()
      .then((mol) =>
        Promise.all([
          mol,
          fetch(structureUrl(root)).then((response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            return response.text()
          }),
        ]),
      )
      .then(([mol, data]) => {
        const viewer = mol.createViewer(root, {
          antialias: true,
          backgroundAlpha: 0,
          preserveDrawingBuffer: true,
        })
        viewer.addModel(data, "pdb")
        styleModel(viewer)
        viewer.setBackgroundColor(0x000000, 0)
        viewer.zoomTo({})
        viewer.zoom(1.34)
        viewer.rotate(8, { x: 0, y: 1, z: 0 })
        viewer.rotate(-4, { x: 1, y: 0, z: 0 })
        viewer.translate(28, -18)
        viewer.render()
        if (shouldSpin()) viewer.spin("y", 0.18, true)
        root.dataset.mmMolStatus = "ready"
        root.mmViewer = viewer
        hero && hero.classList.add("mm-hero--mol-ready")
      })
      .catch(() => {
        molPromise = undefined
        delete root.dataset.mmMolStatus
        hero && hero.classList.remove("mm-hero--mol-ready")
        root.replaceChildren()
      })
  }

  function resizeP53Hero() {
    const root = document.querySelector("[data-mm-p53-viewer]")
    if (root && root.mmViewer) {
      root.mmViewer.resize()
      root.mmViewer.render()
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initP53Hero)
  } else {
    initP53Hero()
  }
  if (window.document$) {
    window.document$.subscribe(initP53Hero)
  }
  window.addEventListener("resize", resizeP53Hero)
})()
