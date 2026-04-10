// Deliberately unsafe: SQL string concat (for SAST evaluation only).
function buildQuery(userId) {
  return "SELECT * FROM users WHERE id = '" + userId + "'";
}

module.exports = { buildQuery };
