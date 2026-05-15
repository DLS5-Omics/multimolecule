(() => {
  const selector = ".mm-section--paths";

  function initWorkflowDots(root = document) {
    root.querySelectorAll(selector).forEach((section) => {
      const rail = section.querySelector(".mm-workflow-screens");
      const panels = Array.from(section.querySelectorAll(".mm-workflow-panel"));
      const dots = Array.from(section.querySelectorAll(".mm-workflow-dot"));
      if (
        !rail ||
        panels.length === 0 ||
        panels.length !== dots.length ||
        rail.dataset.mmWorkflowDots
      ) {
        return;
      }
      rail.dataset.mmWorkflowDots = "true";

      const setActive = (index) => {
        dots.forEach((dot, dotIndex) => {
          const active = dotIndex === index;
          dot.classList.toggle("is-active", active);
          if (active) {
            dot.setAttribute("aria-current", "true");
          } else {
            dot.removeAttribute("aria-current");
          }
        });
      };

      const closestPanelIndex = () => {
        const railLeft = rail.getBoundingClientRect().left;
        let index = 0;
        let distance = Number.POSITIVE_INFINITY;
        panels.forEach((panel, panelIndex) => {
          const delta = Math.abs(panel.getBoundingClientRect().left - railLeft);
          if (delta < distance) {
            distance = delta;
            index = panelIndex;
          }
        });
        return index;
      };

      let animationFrame = 0;
      rail.addEventListener(
        "scroll",
        () => {
          if (animationFrame) {
            return;
          }
          animationFrame = window.requestAnimationFrame(() => {
            animationFrame = 0;
            setActive(closestPanelIndex());
          });
        },
        { passive: true },
      );

      dots.forEach((dot, index) => {
        dot.addEventListener("click", (event) => {
          event.preventDefault();
          panels[index].scrollIntoView({
            behavior: "smooth",
            block: "nearest",
            inline: "start",
          });
          setActive(index);
        });
      });

      setActive(closestPanelIndex());
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initWorkflowDots());
  } else {
    initWorkflowDots();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => initWorkflowDots());
  }
})();
