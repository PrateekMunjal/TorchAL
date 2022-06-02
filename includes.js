var includeFiles = require('./scripts/fileInclude.js');
var renderPuml = require('./scripts/pumlpreprocessor.js');

module.exports = (markdown, options) => {
    return new Promise((resolve, reject) => {
        return resolve(
            renderPuml(includeFiles(markdown))
        );
    })
}
