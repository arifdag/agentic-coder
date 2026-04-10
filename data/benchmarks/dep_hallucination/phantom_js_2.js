// Phantom scoped package
const { render } = require('@superlib/renderer');

function draw(data) {
  return render(data);
}
module.exports = { draw };
