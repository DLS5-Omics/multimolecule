window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  startup: {
    typeset: false,
  },
};

document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    MathJax.typesetPromise();
  }
});
