// Verify backpressure and reconnect recovery without requesting a real camera.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let tick;
let now = 100;
const video = { videoWidth: 320 };
const context = {
  console,
  performance: { now: () => now },
  window: { location: { protocol: 'https:', host: 'example.onrender.com' }, addEventListener() {} },
  document: {
    addEventListener() {},
    getElementById: () => video,
    createElement: () => ({ getContext: () => ({ drawImage() {} }), toDataURL: () => 'data:image/jpeg;base64,test' }),
  },
  setInterval: (fn) => { tick = fn; return 1; },
  clearInterval() {},
  setTimeout() {},
  WebSocket: class {
    static OPEN = 1;
    readyState = 1;
    bufferedAmount = 0;
    sent = [];
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; }
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('static/app.js', 'utf8'), context);
vm.runInContext('initWebSocket(); startFrameStreaming(); socket.onopen();', context);
tick();
tick();
assert.equal(vm.runInContext('socket.sent.length', context), 1, 'only one frame may be in flight');
vm.runInContext('handleTelemetry = () => {}; socket.onmessage({data: JSON.stringify({frame_processed: true})});', context);
tick();
assert.equal(vm.runInContext('socket.sent.length', context), 2, 'acknowledgement permits the next frame');
now += 10001;
tick();
assert.equal(vm.runInContext('socket.readyState', context), 3, 'stalled inference triggers reconnect');
vm.runInContext('socket.onclose(); initWebSocket(); socket.onopen();', context);
tick();
assert.equal(vm.runInContext('socket.sent.length', context), 1, 'new connection can send frames');
vm.runInContext('frameInFlight = false; socket.bufferedAmount = 100;', context);
tick();
assert.equal(vm.runInContext('socket.sent.length', context), 1, 'network buffer applies backpressure');
console.log('Frame streaming backpressure and reconnect checks passed.');
