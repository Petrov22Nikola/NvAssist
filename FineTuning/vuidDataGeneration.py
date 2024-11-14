import os
import re
import json

# VUIDS
vuidFilePath = fr'C:\VVL\layers\vulkan\generated\vk_validation_error_messages.h'
# Source
sourceDirectory = fr'C:\VVL\layers'
# Datasets
trainOutputFilePath = 'train_chatml.json'
evalOutputFilePath = 'eval_chatml.json'

# Read VUIDS
def readVuids(vuidFile):
    vuids = []
    vuidPattern = re.compile(r'\{"(VUID-[^"]+)", "([^"]+)", "([^"]+)"\}')
    with open(vuidFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            match = vuidPattern.search(line)
            if match:
                fullVuidEntry = match.group(0)
                vuids.append(fullVuidEntry)
    return vuids

# Extract variables
def extractVariables(line):
    varPattern = re.compile(r'\b([a-zA-Z_]\w*)\b')
    return set(varPattern.findall(line))

# Capture context after a matched line (until all braces are closed)
def captureAfterContext(lines, startIndex):
    afterContext = []
    braceCount = 0
    for i in range(startIndex, len(lines)):
        line = lines[i].strip()
        afterContext.append(lines[i])
        braceCount += line.count('{') - line.count('}')
        if braceCount <= 0 and line.endswith(';'):
            break
    return afterContext

# Capture relevant previous lines (including 'if' statements and variable definitions)
def captureBeforeContext(lines, startIndex, usedVariables):
    beforeContext = []
    for i in range(startIndex - 1, max(0, startIndex - 4), -1):
        line = lines[i].strip()
        if line.startswith('if'):
            beforeContext.insert(0, lines[i])
            usedVariables.update(extractVariables(line))
            break
        definedVars = extractVariables(line)
        if usedVariables & definedVars:
            beforeContext.insert(0, lines[i])
    return beforeContext

# Unmatched braces
def hasUnmatchedBrace(context):
    openBraces = context.count('{')
    closeBraces = context.count('}')
    return openBraces > closeBraces

# Close brace
def closeUnmatchedBrace(context):
    if hasUnmatchedBrace(context):
        context = context.rstrip() + '\n}'
    return context

# Normalize indentation
def normalizeIndentation(contextLines):
    if not contextLines:
        return contextLines
    firstLineIndentation = len(contextLines[0]) - len(contextLines[0].lstrip())
    normalizedLines = [contextLines[0].lstrip()]
    for line in contextLines[1:]:
        strippedLine = line.lstrip()
        normalizedLine = ' ' * max(0, len(line) - len(line.lstrip()) - firstLineIndentation) + strippedLine
        normalizedLines.append(normalizedLine)
    return normalizedLines

# Search for VUID implementations in source files and capture context
def findVuidContext(sourceDir, vuids):
    vuidImplementations = {}
    
    vuidIdentifiers = [re.search(r'"(VUID-[^"]+)"', vuid).group(1) for vuid in vuids]
    excludePattern = re.compile(r'vuid_\d+\s*=\s*"VUID-[^"]+"')

    for root, _, files in os.walk(sourceDir):
        for file in files:
            if file.endswith(('.cpp', '.h')):
                filePath = os.path.join(root, file)
                if (filePath == vuidFilePath or "vuid" in filePath or "sync" in filePath or "generated" in filePath):
                    continue
                print(filePath)
                with open(filePath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if excludePattern.search(line):  
                            continue

                        for vuidIdentifier in vuidIdentifiers:
                            if vuidIdentifier in line:
                                fullVuidEntry = next(vuid for vuid in vuids if vuidIdentifier in vuid)
                                if fullVuidEntry not in vuidImplementations:
                                    vuidImplementations[fullVuidEntry] = []
                                usedVariables = extractVariables(line)
                                beforeContext = captureBeforeContext(lines, i, usedVariables)
                                afterContext = captureAfterContext(lines, i)
                                fullContext = beforeContext + afterContext
                                normalizedContext = normalizeIndentation(fullContext)
                                fullContextStr = ''.join(normalizedContext)
                                fullContextStrWithBraceCheck = closeUnmatchedBrace(fullContextStr)
                                
                                vuidImplementations[fullVuidEntry].append({
                                    'file': filePath,
                                    'lineNumber': i + 1,
                                    'context': fullContextStrWithBraceCheck,
                                })
    
    return vuidImplementations

vuids = readVuids(vuidFilePath)
vuidContexts = findVuidContext(sourceDirectory, vuids)

train_chatml_data = {"messages": []}
eval_chatml_data = {"messages": []}
counter = 0

for vuid, details in vuidContexts.items():
    for detail in details:
        message_pair = [
            {"role": "user", "content": vuid},
            {"role": "assistant", "content": detail["context"]} 
        ]
        
        if counter % 5 == 0:
            eval_chatml_data["messages"].append(message_pair) 
        
        train_chatml_data["messages"].append(message_pair)
        
        counter += 1

with open(trainOutputFilePath, 'w', encoding='utf-8') as train_file:
    json.dump(train_chatml_data, train_file, indent=4)

with open(evalOutputFilePath, 'w', encoding='utf-8') as eval_file:
    json.dump(eval_chatml_data, eval_file, indent=4)

print(f"Training ChatML data written to {trainOutputFilePath}")
print(f"Evaluation ChatML data written to {evalOutputFilePath}")