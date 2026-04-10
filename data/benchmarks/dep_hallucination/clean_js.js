// All valid imports -- should pass cleanly
const path = require('path');
const fs = require('fs');

function readConfig(dir) {
  return JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8'));
}
module.exports = { readConfig };
