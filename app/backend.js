/* Where the compute lives, and where the page lives.
 *
 * Two rewrites, both because the app was written to be served from the root of
 * its own server and is now served from a subpath of a static site with the
 * compute somewhere else entirely:
 *
 *   /demos/...  the cherrypicked audio, alongside the page, wherever that is
 *   /api, /ws   the matching backend, a machine at home behind a tunnel
 *
 * Doing it here rather than at each call site keeps app.js working unchanged
 * against a local `python app/server.py`, where both resolve to the origin.
 *
 * It also gives the page one honest thing to say when that machine is off: a
 * failed backend request sets body.backend-offline and the stylesheet takes it
 * from there. The demos are static and keep playing either way.
 */
(function () {
  const API = (window.INSTRUMENTAL_API || '').replace(/\/$/, '');
  // The directory this page is served from, e.g. "/instrumental/" or "/".
  const BASE = location.pathname.replace(/[^/]*$/, '');

  const resolve = (url) => {
    if (typeof url !== 'string') return { url, backend: false };
    if (url.startsWith('/demos/')) return { url: BASE + url.slice(1), backend: false };
    if (API && (url.startsWith('/api/'))) return { url: API + url, backend: true };
    return { url, backend: false };
  };

  const offline = (yes) => document.body.classList.toggle('backend-offline', yes);

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const original = typeof input === 'string' ? input : input.url;
    const { url, backend } = resolve(original);
    if (url !== original) {
      input = typeof input === 'string' ? url : new Request(url, input);
    }
    try {
      const res = await nativeFetch(input, init);
      if (backend) offline(false);
      return res;
    } catch (err) {
      if (backend) offline(true);
      throw err;
    }
  };

  // Audio never touches fetch, and it arrives two ways: `new Audio(src)` for
  // the demo players, and `el.src = ...` for the separated stem in the result
  // view. The second one is why the Original lane was silent on the deployed
  // site: the stem URL stayed same-origin and 404d against the static host,
  // while the matched lane played because it is a blob from a rewritten fetch.
  const NativeAudio = window.Audio;
  window.Audio = function (src) {
    return src === undefined ? new NativeAudio() : new NativeAudio(resolve(src).url);
  };
  window.Audio.prototype = NativeAudio.prototype;

  const srcProp = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'src');
  Object.defineProperty(HTMLMediaElement.prototype, 'src', {
    configurable: true,
    enumerable: srcProp.enumerable,
    get() { return srcProp.get.call(this); },
    set(value) { srcProp.set.call(this, resolve(value).url); },
  });

  const NativeWS = window.WebSocket;
  window.WebSocket = function (url, protocols) {
    if (API && typeof url === 'string' && /^wss?:\/\/[^/]+\/ws\//.test(url)) {
      const host = API.replace(/^https?:\/\//, '');
      const proto = API.startsWith('https') ? 'wss://' : 'ws://';
      url = proto + host + url.replace(/^wss?:\/\/[^/]+/, '');
    }
    return new NativeWS(url, protocols);
  };
  window.WebSocket.prototype = NativeWS.prototype;

  // Ask once on load, so the page is honest before anyone clicks.
  if (API) {
    nativeFetch(API + '/api/health')
      .then((r) => offline(!r.ok))
      .catch(() => offline(true));
  }
})();
