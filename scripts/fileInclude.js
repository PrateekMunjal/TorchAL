var fs = require('fs');

module.exports = (markdown) => {
  return markdown
    .split('\n')
    .map((line, index) => {
      if(/^{{.*}}$/.test(line))
        return readIncludeFile(line)
      return line
    })
    .join('\n')
}

const readIncludeFile = includeCommand => {
    let fileToRead = includeCommand.replace("{{", "").replace("}}", "");
    return fs.readFileSync(fileToRead, 'utf8')
}
