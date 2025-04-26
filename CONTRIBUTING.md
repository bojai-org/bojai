
# Contribution Guidelines

Thank you for your interest in contributing to **BojAI**!   
We welcome all contributions, especially new **built-in model pipelines**.

If you’ve developed a model you think others could benefit from, follow these steps to add it as a new built-in BojAI pipeline:



## Steps to Contribute a New Built-in Model

1. **Start a new pipeline**
   ```bash
   bojai create --pipeline your-custom-model-name
   ```

2. **Implement and test your pipeline**  
   Make sure it runs cleanly through the BojAI interface and CLI.

3. **Write a model card**  
   Follow the same format as our [pre-built pipelines cards](../pre-built-pipelines/). 

4. **Fork the GitHub repo**  
   [https://github.com/soghomon-b/bojai](https://github.com/soghomon-b/bojai)

5. **Create a new branch**
   ```bash
   git checkout -b your-model-name
   ```

6. **Add your pipeline to the repo**  
   Move your pipeline folder into the `pbm` directory, and rename it:
   ```
   pbm/pbm-your-model-name
   ```

7. **Commit and push your changes**
   ```bash
   git add .
   git commit -m "Add custom pipeline: your-model-name"
   git push origin your-model-name
   ```

8. **Open a pull request**  
   Go to your forked GitHub repo and submit a PR to the main BojAI repository.

9. **We’ll take it from there!**  
   The BojAI team will:
   - Review your code
   - Add a documentation card
   - Merge it into the official release 🎉

---

Thank you for helping us make machine learning easier and more accessible. We can't wait to see what you build!
