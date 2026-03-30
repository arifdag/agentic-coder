/**
 * Sample JavaScript functions for testing the Jest test generation pipeline.
 */

function add(a, b) {
  return a + b;
}

function factorial(n) {
  if (n < 0) throw new Error("Negative numbers not allowed");
  if (n === 0 || n === 1) return 1;
  return n * factorial(n - 1);
}

function isPalindrome(str) {
  if (typeof str !== "string") return false;
  const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, "");
  return cleaned === cleaned.split("").reverse().join("");
}

function findMax(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  return Math.max(...arr);
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

module.exports = { add, factorial, isPalindrome, findMax, capitalize };
