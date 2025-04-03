# Optimizing GPT-Based MCC Classification

This document provides guidance on how to optimize GPT prompts for more accurate Merchant Category Code (MCC) classification.

## Key Factors for Prompt Optimization

### 1. Prompt Structure

The structure of the prompt significantly impacts the quality and consistency of GPT responses. Our current implementation includes:

```
Task: Analyze a merchant name and classify it with the most appropriate Merchant Category Code (MCC).

Merchant: "[merchant_name]"
Legal Name: "[legal_name]" (if available)

Instructions:
1. Carefully analyze the merchant name to determine their primary business.
2. Consider both explicit and implicit business indicators in the name.
3. Select the most appropriate MCC from the list below.
4. If the merchant name is generic or ambiguous, look for subtle clues or patterns.
5. For businesses that could fit multiple categories, select the most specific and probable MCC.
6. Consider common naming patterns for different business types.

Reference MCCs:
- 5411: Grocery Stores, Supermarkets
- 5814: Fast Food Restaurants
...etc...

Respond in this exact format:
1. Analysis: Brief analysis of the merchant name and what it suggests
2. Primary MCC: [MCC code]
3. Description: [MCC description]
4. Confidence: [High/Medium/Low] with explanation
5. Alternative MCCs: List 2-3 other possible MCCs in order of likelihood
```

### 2. MCC Selection in Prompt

Including all MCCs would create an excessively long prompt, so we need to be strategic about which MCCs to include:

- **Common MCCs**: Include the most frequently used MCCs across different merchant types
- **Relevant Categories**: Group MCCs by industry sector to ensure coverage of major categories
- **Specificity**: Include both general and specific MCCs to allow for proper classification granularity

### 3. System Message Optimization

The system message sets the tone and expertise level for the GPT model. Our current implementation uses:

```
You are an expert in merchant classification and MCC codes with extensive knowledge of business types and industries.
```

Consider enhancing this with:

```
You are an expert in merchant classification and MCC codes with extensive knowledge of business types and industries. Your task is to accurately determine the most appropriate MCC code based on merchant names, using pattern recognition and industry knowledge. Be specific in your classifications, and consider both explicit and implicit business indicators in merchant names.
```

### 4. Temperature Setting

A lower temperature setting (0.3 in our implementation) produces more consistent and deterministic results, which is ideal for classification tasks.

## Prompt Tuning Techniques

### 1. Few-Shot Learning

Include examples of successful classifications to help the model understand the expected reasoning pattern:

```
Example 1:
Merchant: "Joe's Pizza"
Analysis: The name clearly indicates this is a pizza restaurant.
Primary MCC: 5812
Description: Eating places and Restaurants
Confidence: High - The merchant name explicitly states it's a pizza establishment.

Example 2:
Merchant: "Tech Solutions"
Analysis: The name suggests a technology services or consulting business.
Primary MCC: 7372
Description: Computer Programming, Integrated Systems Design and Data Processing Services
Confidence: Medium - While technology-focused, the exact nature of services is somewhat ambiguous.
```

### 2. Progressive Refinement Testing

Test your prompt with a diverse set of merchant names and iteratively refine based on results:

1. Start with obvious merchant names (e.g., "City Grocery", "Downtown Dental Clinic")
2. Test with ambiguous names (e.g., "John Smith", "Superior Services")
3. Test with names that could fit multiple categories (e.g., "The Corner Store")
4. Test with industry-specific jargon (e.g., "Cloud DevOps Solutions")

### 3. Response Format Enforcement

Use explicit instructions for response formatting to ensure consistency:

- Emphasize the exact format required
- Make it clear that all sections must be completed
- Specify how confidence levels should be determined

## Common Challenges and Solutions

### 1. Ambiguous Merchant Names

For merchants with generic names like "John Smith" or "Quality Services":

- Instruct the model to look for subtle clues in naming patterns
- Have the model provide lower confidence scores for ambiguous cases
- Prioritize general service categories when specific indicators are lacking

### 2. Multi-Category Merchants

For merchants that could fit multiple categories:

- Provide guidance on how to prioritize the primary business function
- Instruct the model to use the alternative MCCs section for secondary categories
- Include rules for tie-breaking when multiple categories seem equally likely

### 3. Regional/Cultural Naming Patterns

Different regions may have distinct naming patterns:

- Include instructions to consider regional naming conventions
- Test with merchant names from various regions to ensure adaptability

## Evaluation and Iteration

Continually evaluate classification performance:

1. Track accuracy metrics for different merchant types
2. Identify categories where misclassifications are common
3. Adjust the prompt with specific guidance for problematic categories
4. Consider A/B testing different prompt variations to determine which performs best

## Model Considerations

Different GPT models may require different prompt optimizations:

- GPT-4 performs better with nuanced instructions and complex reasoning
- GPT-3.5 may benefit from more explicit examples and direct instructions
- Consider model-specific prompt versions if using multiple models

By following these guidelines and continuously refining the approach, you can achieve significant improvements in MCC classification accuracy. 