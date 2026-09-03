(function () {
  var targets = document.querySelectorAll("[data-reveal]");
  if (!targets.length) return;

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) {
    targets.forEach(function (el) { el.classList.add("is-visible"); });
    return;
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  targets.forEach(function (el) { observer.observe(el); });
})();

// Email: the address is assembled here from two data attributes, so it never
// sits in the HTML as one string for scrapers. Humans get a normal mailto link
// whose visible text is the address itself.
(function () {
  var links = document.querySelectorAll("a.email-link[data-u][data-d]");
  Array.prototype.forEach.call(links, function (a) {
    var addr = a.getAttribute("data-u") + "\u0040" + a.getAttribute("data-d");
    a.setAttribute("href", "mailto:" + addr);
    a.textContent = addr;
  });
})();

// Table of contents: generated from the post body's own headings, plus
// hover-visible anchor links on each heading and a scroll-spy highlight.
(function () {
  var body = document.querySelector("[data-post-body]");
  var toc = document.querySelector("[data-toc]");
  var list = document.querySelector("[data-toc-list]");
  if (!body || !toc || !list) return;

  var headings = Array.prototype.slice.call(body.querySelectorAll("h2, h3"));
  if (headings.length < 2) {
    toc.setAttribute("data-empty", "");
    return;
  }

  var links = [];
  headings.forEach(function (h) {
    if (!h.id) return;

    var anchor = document.createElement("a");
    anchor.className = "heading-anchor";
    anchor.href = "#" + h.id;
    anchor.setAttribute("aria-label", "Link to this section");
    anchor.textContent = "#";
    h.appendChild(anchor);

    var li = document.createElement("li");
    if (h.tagName === "H3") li.className = "toc-h3";
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/#$/, "").trim();
    li.appendChild(a);
    list.appendChild(li);
    links.push({ id: h.id, el: a });
  });

  toc.setAttribute("data-ready", "");

  if (!("IntersectionObserver" in window) || !links.length) return;

  var byId = {};
  links.forEach(function (l) { byId[l.id] = l.el; });

  var spy = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        var link = byId[entry.target.id];
        if (!link) return;
        if (entry.isIntersecting) {
          links.forEach(function (l) { l.el.classList.remove("is-active"); });
          link.classList.add("is-active");
        }
      });
    },
    { rootMargin: "-15% 0px -70% 0px" }
  );
  headings.forEach(function (h) { if (h.id) spy.observe(h); });
})();

// Wrap tables so wide ones scroll horizontally on narrow screens instead of
// breaking the layout, without disturbing the breakout-grid CSS targets it.
(function () {
  var body = document.querySelector("[data-post-body]");
  if (!body) return;
  Array.prototype.slice.call(body.querySelectorAll(":scope > table")).forEach(function (table) {
    var wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
})();

// Tag filtering on /writing/ — client-side, shareable via #tag in the URL.
(function () {
  var bar = document.querySelector("[data-tag-filter]");
  if (!bar) return;

  var chips = Array.prototype.slice.call(bar.querySelectorAll(".tag-chip"));
  var items = Array.prototype.slice.call(document.querySelectorAll(".writing-list li[data-tags]"));
  var years = Array.prototype.slice.call(document.querySelectorAll(".archive-year"));
  var noResults = document.querySelector("[data-no-results]");

  function applyFilter(tag) {
    var visibleCount = 0;
    items.forEach(function (li) {
      var tags = (li.getAttribute("data-tags") || "").split(" ");
      var show = !tag || tags.indexOf(tag) !== -1;
      li.hidden = !show;
      if (show) visibleCount++;
    });
    years.forEach(function (section) {
      section.hidden = section.querySelectorAll("li[data-tags]:not([hidden])").length === 0;
    });
    if (noResults) noResults.hidden = visibleCount > 0;
    chips.forEach(function (chip) {
      chip.classList.toggle("is-active", chip.getAttribute("data-tag-value") === (tag || ""));
    });
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      var tag = chip.getAttribute("data-tag-value");
      applyFilter(tag);
      var url = tag ? "#" + encodeURIComponent(tag) : location.pathname + location.search;
      history.replaceState(null, "", url);
    });
  });

  function applyFromHash() {
    var tag = location.hash ? decodeURIComponent(location.hash.slice(1)) : "";
    if (!tag || chips.some(function (c) { return c.getAttribute("data-tag-value") === tag; })) {
      applyFilter(tag);
    }
  }

  // Individual tags in the list (and on post pages) are plain <a href="#tag">
  // links, not chip buttons — this catches those clicks too, since they only
  // change the URL hash rather than triggering a full navigation on this page.
  window.addEventListener("hashchange", applyFromHash);
  applyFromHash();
})();
