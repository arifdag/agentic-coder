const leftpadx = require("leftpadx");

function pad(s, n) {
  return leftpadx(s, n);
}

module.exports = { pad };
