#pragma once

// Simple YAML parser for basic use cases
// This is a lightweight implementation focused on our specific needs

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace YAML {

// Exception class for YAML parsing errors
class Exception : public std::runtime_error {
public:
    Exception(const std::string& message) : std::runtime_error(message) {}
};

class Node;
using NodePtr = std::shared_ptr<Node>;

enum class NodeType {
    Null,
    Scalar,
    Sequence,
    Map
};

class Node {
public:
    NodeType m_type = NodeType::Null;
    std::string m_scalar;
    std::vector<NodePtr> m_sequence;
    std::map<std::string, NodePtr> m_map;
    
    Node() = default;
    Node(const std::string& value) : m_type(NodeType::Scalar), m_scalar(value) {}
    
    NodeType Type() const { return m_type; }
    
    // Scalar access
    template<typename T>
    T as() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return m_scalar;
        } else if constexpr (std::is_same_v<T, int>) {
            return std::stoi(m_scalar);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(m_scalar);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::stod(m_scalar);
        } else if constexpr (std::is_same_v<T, bool>) {
            std::string lower = m_scalar;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            return lower == "true" || lower == "1" || lower == "yes";
        }
        return T{};
    }
    
    // Map access
    Node operator[](const std::string& key) const {
        auto it = m_map.find(key);
        return (it != m_map.end()) ? *it->second : Node();
    }
    
    Node operator[](const char* key) const {
        return (*this)[std::string(key)];
    }
    
    // Sequence access
    Node operator[](std::size_t index) const {
        return (index < m_sequence.size()) ? *m_sequence[index] : Node();
    }
    
    // Existence checks
    bool IsDefined() const { return m_type != NodeType::Null; }
    bool IsScalar() const { return m_type == NodeType::Scalar; }
    bool IsSequence() const { return m_type == NodeType::Sequence; }
    bool IsMap() const { return m_type == NodeType::Map; }
    
    // Boolean conversion operator for conditional expressions
    explicit operator bool() const { return IsDefined(); }
    
    // Size
    size_t size() const {
        if (m_type == NodeType::Sequence) return m_sequence.size();
        if (m_type == NodeType::Map) return m_map.size();
        return 0;
    }
    
    // Simple iteration support for range-based for loops
    std::vector<NodePtr>::const_iterator begin() const {
        return m_sequence.begin();
    }
    
    std::vector<NodePtr>::const_iterator end() const {
        return m_sequence.end();
    }
};

class Parser {
private:
    std::vector<std::string> m_lines;
    size_t m_currentLine = 0;
    
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }
    
    static int getIndentLevel(const std::string& line) {
        int indent = 0;
        for (char c : line) {
            if (c == ' ') indent++;
            else break;
        }
        return indent;
    }
    
    static bool isListItem(const std::string& line) {
        std::string trimmed = trim(line);
        return trimmed.length() >= 2 && trimmed.substr(0, 2) == "- ";
    }
    
    static std::pair<std::string, std::string> parseKeyValue(const std::string& line) {
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            return {"", ""};
        }
        
        std::string key = trim(line.substr(0, colonPos));
        std::string value = trim(line.substr(colonPos + 1));
        
        // Remove quotes if present
        if (value.length() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        return {key, value};
    }
    
    NodePtr parseValue(int baseIndent) {
        if (m_currentLine >= m_lines.size()) {
            return std::make_shared<Node>();
        }
        
        std::string currentLine = m_lines[m_currentLine];
        int currentIndent = getIndentLevel(currentLine);
        std::string trimmed = trim(currentLine);
        
        // Check if this is a list
        if (isListItem(trimmed)) {
            return parseSequence(baseIndent);
        }
        
        // Check if next lines have higher indent (indicating a map)
        if (m_currentLine + 1 < m_lines.size()) {
            int nextIndent = getIndentLevel(m_lines[m_currentLine + 1]);
            if (nextIndent > currentIndent) {
                return parseMap(baseIndent);
            }
        }
        
        // It's a scalar value after the colon
        auto [key, value] = parseKeyValue(trimmed);
        m_currentLine++;
        return std::make_shared<Node>(value);
    }
    
    NodePtr parseSequence(int baseIndent) {
        auto node = std::make_shared<Node>();
        node->m_type = NodeType::Sequence;
        
        while (m_currentLine < m_lines.size()) {
            std::string currentLine = m_lines[m_currentLine];
            int currentIndent = getIndentLevel(currentLine);
            std::string trimmed = trim(currentLine);
            
            // If indent is less than base, we're done with this sequence
            if (currentIndent < baseIndent) {
                break;
            }
            
            // If it's not a list item at this level, we're done
            if (currentIndent == baseIndent && !isListItem(trimmed)) {
                break;
            }
            
            if (isListItem(trimmed) && currentIndent == baseIndent) {
                // Parse list item
                std::string itemContent = trimmed.substr(2); // Remove "- "
                
                if (itemContent.empty()) {
                    // Multi-line list item
                    m_currentLine++;
                    node->m_sequence.push_back(parseValue(baseIndent + 2));
                } else {
                    // Inline list item
                    auto [key, value] = parseKeyValue(itemContent);
                    if (!key.empty() && !value.empty()) {
                        // It's a key-value pair in the list item
                        auto itemNode = std::make_shared<Node>();
                        itemNode->m_type = NodeType::Map;
                        itemNode->m_map[key] = std::make_shared<Node>(value);
                        
                        m_currentLine++;
                        
                        // Check for additional properties of this list item
                        while (m_currentLine < m_lines.size()) {
                            int nextIndent = getIndentLevel(m_lines[m_currentLine]);
                            if (nextIndent <= baseIndent) break;
                            
                            std::string nextTrimmed = trim(m_lines[m_currentLine]);
                            if (isListItem(nextTrimmed)) break;
                            
                            auto [nextKey, nextValue] = parseKeyValue(nextTrimmed);
                            if (!nextKey.empty()) {
                                if (!nextValue.empty()) {
                                    itemNode->m_map[nextKey] = std::make_shared<Node>(nextValue);
                                    m_currentLine++;
                                } else {
                                    // This key has a nested structure - check if next line has higher indent
                                    m_currentLine++;
                                    if (m_currentLine < m_lines.size()) {
                                        int nestedIndent = getIndentLevel(m_lines[m_currentLine]);
                                        if (nestedIndent > nextIndent) {
                                            // Parse as nested map
                                            itemNode->m_map[nextKey] = parseMap(nestedIndent);
                                        } else {
                                            // Empty value
                                            itemNode->m_map[nextKey] = std::make_shared<Node>();
                                        }
                                    } else {
                                        itemNode->m_map[nextKey] = std::make_shared<Node>();
                                    }
                                    continue;
                                }
                            } else {
                                m_currentLine++;
                            }
                        }
                        
                        node->m_sequence.push_back(itemNode);
                    } else {
                        // Simple scalar list item
                        node->m_sequence.push_back(std::make_shared<Node>(itemContent));
                        m_currentLine++;
                    }
                }
            } else {
                m_currentLine++;
            }
        }
        
        return node;
    }
    
    NodePtr parseMap(int baseIndent) {
        auto node = std::make_shared<Node>();
        node->m_type = NodeType::Map;
        
        while (m_currentLine < m_lines.size()) {
            std::string currentLine = m_lines[m_currentLine];
            int currentIndent = getIndentLevel(currentLine);
            std::string trimmed = trim(currentLine);
            
            // If indent is less than base, we're done with this map
            if (currentIndent < baseIndent) {
                break;
            }
            
            // Skip lines that are too indented for this level
            if (currentIndent > baseIndent) {
                m_currentLine++;
                continue;
            }
            
            auto [key, value] = parseKeyValue(trimmed);
            if (key.empty()) {
                m_currentLine++;
                continue;
            }
            
            if (!value.empty()) {
                // Simple key-value pair
                node->m_map[key] = std::make_shared<Node>(value);
                m_currentLine++;
            } else {
                // Key with nested structure
                m_currentLine++;
                node->m_map[key] = parseValue(baseIndent + 2);
            }
        }
        
        return node;
    }
    
public:
    NodePtr parse(const std::string& content) {
        m_lines.clear();
        m_currentLine = 0;
        
        // Split content into lines
        std::istringstream stream(content);
        std::string line;
        while (std::getline(stream, line)) {
            m_lines.push_back(line);
        }
        
        // Remove empty lines and comments
        m_lines.erase(
            std::remove_if(m_lines.begin(), m_lines.end(), [](const std::string& line) {
                std::string trimmed = trim(line);
                return trimmed.empty() || trimmed[0] == '#';
            }),
            m_lines.end()
        );
        
        if (m_lines.empty()) {
            return std::make_shared<Node>();
        }
        
        return parseValue(0);
    }
};

// Main interface functions
inline Node LoadFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Node();
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    Parser parser;
    auto nodePtr = parser.parse(content);
    return nodePtr ? *nodePtr : Node();
}

inline Node Load(const std::string& content) {
    Parser parser;
    auto nodePtr = parser.parse(content);
    return nodePtr ? *nodePtr : Node();
}

} // namespace YAML