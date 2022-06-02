var fs = require('fs');

module.exports = (markdown) => {
    let pumldiagram = "";
    let is_within_puml_block = false;
    return markdown
        .split('\n')
        .map((line, index) => {
            const is_current_puml_line_start = "```puml" == line;
            const is_current_puml_line_end = is_within_puml_block && "```" == line;
            if (!is_within_puml_block && !is_current_puml_line_start)
                return line;

            if (is_within_puml_block && /^!include.*$/.test(line)) {
                pumldiagram += readIncludeFile(line.replace("!include ", ""));
                return "<!-- emptyline -->";
            }

            if (is_current_puml_line_start)
                is_within_puml_block = true;

            if (is_current_puml_line_end) {
                is_within_puml_block = false;
                pumlImgUrl = "<img src='https://g.gravizo.com/svg?@startuml;" + pumldiagram + "@enduml'/>";
                pumldiagram = "";
                return pumlImgUrl
            }
            if (is_within_puml_block && !is_current_puml_line_start)
                pumldiagram += line.trim() + ";";

            return "<!-- emptyline -->";
        })
        .filter(value => value != "<!-- emptyline -->")
        .join('\n')
}

const readIncludeFile = filename => {
    return fs.readFileSync(filename, 'utf8')
        .split('\n')
        .map((line, index) => {
            if(line.trim().startsWith("'"))
                return "<!-- emptyline -->";
            if (line.includes("'"))
                return line.substr(0, line.indexOf("'"));
            return line
            })
        .map((line, index) => {line.replace("#", "")})
        .filter(value => value != "<!-- emptyline -->")
        .join(';')
}
