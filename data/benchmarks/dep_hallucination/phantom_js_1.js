// Phantom npm package alongside a valid one
const express = require('express');
const ultravalidator = require('ultravalidator');

function handle(req) {
  return ultravalidator.check(req.body);
}
module.exports = { handle };
