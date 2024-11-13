import os
import re
import json

# Path to the file containing the list of VUIDs
vuidFilePath = fr'C:\VVL\layers\vulkan\generated\vk_validation_error_messages.h'
# Path to the directory containing the Vulkan source files
sourceDirectory = fr'C:\VVL\layers'
# Output paths for chatml.json files
trainOutputFilePath = 'train_chatml.json'
evalOutputFilePath = 'eval_chatml.json'

# Function to read VUIDs from vk_validation_error_messages.h
def readVuids(vuidFile):
    vuids = []
    vuidPattern = re.compile(r'\{"(VUID-[^"]+)", "([^"]+)", "([^"]+)"\}')
    with open(vuidFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            match = vuidPattern.search(line)
            if match:
                fullVuidEntry = match.group(0)  # Capture the entire line as the VUID entry
                vuids.append(fullVuidEntry)
    return vuids

# Function to extract variable names from a line of code
def extractVariables(line):
    varPattern = re.compile(r'\b([a-zA-Z_]\w*)\b')
    return set(varPattern.findall(line))

# Function to capture context after a matched line (until all braces are closed)
def captureAfterContext(lines, startIndex):
    afterContext = []
    braceCount = 0  # To track opening and closing braces
    # Traverse forwards starting from the current line (including it)
    for i in range(startIndex, len(lines)):
        line = lines[i].strip()
        afterContext.append(lines[i])
        # Track opening and closing braces to detect when an 'if' block ends
        braceCount += line.count('{') - line.count('}')
        # Stop once all opened braces are closed or we encounter a semicolon
        if braceCount <= 0 and line.endswith(';'):
            break
    return afterContext

# Function to capture relevant previous lines (including 'if' statements and variable definitions)
def captureBeforeContext(lines, startIndex, usedVariables):
    beforeContext = []
    # Traverse backwards from the matched line
    for i in range(startIndex - 1, max(0, startIndex - 4), -1):
        line = lines[i].strip()
        # Check if it's an 'if' statement and include it
        if line.startswith('if'):
            beforeContext.insert(0, lines[i])  # Add 'if' statement to context
            usedVariables.update(extractVariables(line))  # Extract variables from 'if' condition
            break
        # Check if this line defines any of the used variables (e.g., int x = 5;)
        definedVars = extractVariables(line)
        if usedVariables & definedVars:  # If there's an intersection with used variables
            beforeContext.insert(0, lines[i])  # Include this line in context
    return beforeContext

# Function to check if there are unmatched opening braces in the context
def hasUnmatchedBrace(context):
    openBraces = context.count('{')
    closeBraces = context.count('}')
    return openBraces > closeBraces

# Function to close unmatched opening braces by appending a closing brace at the end without extra empty lines
def closeUnmatchedBrace(context):
    if hasUnmatchedBrace(context):
        # Remove any trailing whitespace or newlines before appending the closing brace
        context = context.rstrip() + '\n}'
    return context

# Function to normalize indentation based on the first line's indentation
def normalizeIndentation(contextLines):
    if not contextLines:
        return contextLines
    # Get the indentation level of the first non-empty line
    firstLineIndentation = len(contextLines[0]) - len(contextLines[0].lstrip())
    # Normalize subsequent lines to match this indentation level
    normalizedLines = [contextLines[0].lstrip()]  # Remove leading spaces from first line
    for line in contextLines[1:]:
        strippedLine = line.lstrip()  # Remove leading spaces/tabs
        normalizedLine = ' ' * max(0, len(line) - len(line.lstrip()) - firstLineIndentation) + strippedLine
        normalizedLines.append(normalizedLine)
    return normalizedLines

# Function to search for VUID implementations in source files and capture context (now excludes vuid_number assignments)
def findVuidContext(sourceDir, vuids):
    vuidImplementations = {}
    
    vuidIdentifiers = [re.search(r'"(VUID-[^"]+)"', vuid).group(1) for vuid in vuids]  # Extract just VUID identifiers
    
    # Regex pattern to exclude lines like: vuid_XXXXXX = "VUID-..."
    excludePattern = re.compile(r'vuid_\d+\s*=\s*"VUID-[^"]+"')

    # Walk through all files in the directory
    for root, _, files in os.walk(sourceDir):
        for file in files:
            if file.endswith(('.cpp', '.h')):  # Only process .cpp and .h files
                filePath = os.path.join(root, file)
                if (filePath == vuidFilePath or "vuid" in filePath or "sync" in filePath or "generated" in filePath):
                    continue
                print(filePath)
                with open(filePath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Search each line for any VUID reference (not just skip |=)
                    for i, line in enumerate(lines):
                        if excludePattern.search(line):  
                            continue  # Skip lines that match the exclusion pattern (vuid_number assignments)

                        for vuidIdentifier in vuidIdentifiers:
                            if vuidIdentifier in line:  # If a VUID is found in this line and it's not excluded by pattern
                                fullVuidEntry = next(vuid for vuid in vuids if vuidIdentifier in vuid)
                                if fullVuidEntry not in vuidImplementations:
                                    vuidImplementations[fullVuidEntry] = []
                                # Extract variables used in this line
                                usedVariables = extractVariables(line)
                                # Capture context before and after the matched line (including closing brace)
                                beforeContext = captureBeforeContext(lines, i, usedVariables)
                                afterContext = captureAfterContext(lines, i)
                                # Combine before and after context with the matched line itself
                                fullContext = beforeContext + afterContext
                                # Normalize indentation for all lines based on the first line's indentation
                                normalizedContext = normalizeIndentation(fullContext)
                                # Join all lines into a single string for output and check for unmatched braces
                                fullContextStr = ''.join(normalizedContext)
                                fullContextStrWithBraceCheck = closeUnmatchedBrace(fullContextStr)
                                
                                vuidImplementations[fullVuidEntry].append({
                                    'file': filePath,
                                    'lineNumber': i + 1,
                                    'context': fullContextStrWithBraceCheck,
                                })
    
    return vuidImplementations

# Step 1: Extract full VUID entries from vk_validation_error_messages.h (not just identifiers)
vuids = readVuids(vuidFilePath)

# Step 2: Capture VUID context from the source files where any reference to a VUID is found (excluding assignments like vuid_number)
vuidContexts = findVuidContext(sourceDirectory, vuids)

# Step 3: Convert results to chatml.json format (split between train and eval)
train_chatml_data = {"messages": []}
eval_chatml_data = {"messages": []}
index_counter = 0

for vuid, details in vuidContexts.items():
    for detail in details:
        message_pair = [
            {"role": "user", "content": vuid},   # Insert entire VUID entry here as user prompt.
            {"role": "assistant", "content": detail["context"]}   # Insert implementation here as assistant response.
        ]
        
        if index_counter % 5 == 0:
            eval_chatml_data["messages"].append(message_pair)   # Add every fifth pair to eval data.
        
        train_chatml_data["messages"].append(message_pair)   # Add others to train data.
        
        index_counter += 1

# Step 4: Write results to train_chatml.json and eval_chatml.json files 
with open(trainOutputFilePath, 'w', encoding='utf-8') as train_file:
    json.dump(train_chatml_data, train_file, indent=4)

with open(evalOutputFilePath, 'w', encoding='utf-8') as eval_file:
    json.dump(eval_chatml_data, eval_file, indent=4)

print(f"Training ChatML data written to {trainOutputFilePath}")
print(f"Evaluation ChatML data written to {evalOutputFilePath}")