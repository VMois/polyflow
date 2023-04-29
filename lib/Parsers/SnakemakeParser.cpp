#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "Parsers/SnakemakeParser.hpp"


Token SnakemakeLexer::nextToken() {
    skipWhitespace();

    if (input_.eof()) {
        return Token{END_OF_FILE, "END", line_, column_};
    }

    char c = input_.peek();
    switch (c) {
        case ':':
            input_.get();
            return Token{COLON, "COLON", line_, column_++};
        case '\t':
            input_.get();
            return Token{INDENT, "INDENT", line_, column_++};
        case '"':
            return readString();
        case '\n':
            return readNewline();
        case ',':
            input_.get();
            return Token{COMMA, "COMMA", line_, column_++};
        default:
            if (isIdentifierStart(c)) {
                return readIdentifierOrKeyword();
            } else {
                std::cerr << "Unexpected character: " << c
                        << " (line " << line_ << ", column " << column_ << ")\n";
                input_.get();
                return nextToken();
            }
    }
};

bool SnakemakeLexer::isWhitespace(char c) const {
    return c == ' ';
};

void SnakemakeLexer::skipWhitespace() {
    while (isWhitespace(input_.peek())) {
        input_.get();
        column_++;
    }
};

bool SnakemakeLexer::isIdentifierStart(char c) const {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
};

bool SnakemakeLexer::isIdentifierChar(char c) const {
    return isIdentifierStart(c) || (c >= '0' && c <= '9');
};

Token SnakemakeLexer::readString() {
    std::string value;
    input_.get(); // consume the opening quote

    while (input_.peek() != '"') {
        value += input_.get();
        column_++;
    }
    input_.get(); // consume the closing quote
    return Token{STRING, value, line_, column_ - value.length()};
};

Token SnakemakeLexer::readNewline() {
    input_.get(); // consume the newline
    line_++;
    column_ = 0;
    return Token{NEWLINE, "NEW_LINE", line_, column_};
};

Token SnakemakeLexer::readIdentifierOrKeyword() {
    std::string value;
    while (isIdentifierChar(input_.peek())) {
        value += input_.get();
        column_++;
    }

    if (value == "rule") {
        return Token{RULE, value, line_, column_ - value.length()};
    } else if (value == "input") {
        return Token{INPUT, value, line_, column_ - value.length()};
    } else if (value == "output") {
        return Token{OUTPUT, value, line_, column_ - value.length()};
    } else if (value == "shell") {
        return Token{SHELL, value, line_, column_ - value.length()};
    } else {
        return Token{IDENTIFIER, value, line_, column_ - value.length()};
    }
};


void SnakemakeParser::advance() {
    lookahead = lexer_.nextToken();
}

void SnakemakeParser::throw_unexpected_token() {
    throw std::runtime_error("Unexpected token: " + lookahead.value + " on line " + std::to_string(lookahead.line) + ", column " + std::to_string(lookahead.column) + "\n");
}

void SnakemakeParser::match(const TokenType tokenType) {
    if (lookahead.type == tokenType) {
        advance();
    } else {
        throw_unexpected_token();
    }
}

std::shared_ptr<TreeNode> SnakemakeParser::snakemake() {
    if (lookahead.type == END_OF_FILE) return nullptr;

    std::shared_ptr<TreeNode> node;
    if (lookahead.type == RULE) {
        node = rule();
    } else if (lookahead.type == NEWLINE) {
        match(NEWLINE);
    } else {
        throw_unexpected_token();
    }

    if (node) {
        root->add_child(node);
    }

    snakemake();
    return root;
}

std::shared_ptr<TreeNode> SnakemakeParser::rule() {
    match(RULE);

    std::shared_ptr<TreeNode> rule_node = std::make_shared<TreeNode>("rule");

    if (lookahead.type == IDENTIFIER) {
        rule_node->add_child(std::make_shared<TreeNode>(lookahead.value));
        advance();
    }

    match(COLON);
    match(NEWLINE);

    while (lookahead.type == INPUT || lookahead.type == OUTPUT || lookahead.type == SHELL) {
        if (lookahead.type == INPUT) {
            rule_node->add_child(input_rule());
        } else if (lookahead.type == OUTPUT) {
            rule_node->add_child(output_rule());
        } else if (lookahead.type == SHELL) {
            rule_node->add_child(shell_rule());
        }
    }

    return rule_node;
}

std::shared_ptr<TreeNode> SnakemakeParser::input_rule() {
    match(INPUT);
    match(COLON);
    match(NEWLINE);

    std::shared_ptr<TreeNode> input_node = std::make_shared<TreeNode>("input");
    input_node->add_child(parameter_list());

    return input_node;
}

std::shared_ptr<TreeNode> SnakemakeParser::output_rule() {
    match(OUTPUT);
    match(COLON);
    match(NEWLINE);

    std::shared_ptr<TreeNode> output_node = std::make_shared<TreeNode>("output");
    output_node->add_child(parameter_list());

    return output_node;
}

std::shared_ptr<TreeNode> SnakemakeParser::shell_rule() {
    match(SHELL);
    match(COLON);
    match(NEWLINE);

    std::shared_ptr<TreeNode> shell_node = std::make_shared<TreeNode>("shell");

    shell_node->add_child(std::make_shared<TreeNode>(lookahead.value));
    match(STRING);
    
    match(NEWLINE);

    return shell_node;
}

std::shared_ptr<TreeNode> SnakemakeParser::parameter_list() {
    std::shared_ptr<TreeNode> list_node = std::make_shared<TreeNode>("parameter_list");
    while (lookahead.type == STRING) {
        list_node->add_child(std::make_shared<TreeNode>(lookahead.value));

        match(STRING);
        if (lookahead.type == COMMA) {
            match(COMMA);
        }
        match(NEWLINE);
    }
    return list_node;
}
