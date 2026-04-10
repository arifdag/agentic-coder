// Vulnerable: eval of user input in Node.js
function calculate(expr) {
  return eval(expr);
}
module.exports = { calculate };
