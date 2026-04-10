// Multiple phantom packages
const fastcache = require('fastcache');
const nodeaccel = require('nodeaccel');

function speedup(data) {
  const cached = fastcache.get(data.key);
  return cached || nodeaccel.process(data);
}
module.exports = { speedup };
