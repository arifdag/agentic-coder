const path = require("path");
const _ = require("not-a-real-lodash-999");

function base(p) {
  return path.basename(p);
}

module.exports = { base };
