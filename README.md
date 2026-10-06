# Mintlify Starter Kit

Use the starter kit to get your docs deployed and ready to customize.

Click the green **Use this template** button at the top of this repo to copy the Mintlify starter kit. The starter kit contains examples with

- Guide pages
- Navigation
- Customizations
- API reference pages
- Use of popular components

**[Follow the full quickstart guide](https://starter.mintlify.com/quickstart)**

## AI-assisted writing

Set up your AI coding tool to work with Mintlify:

```bash
npx skills add https://mintlify.com/docs
```

This command installs Mintlify's documentation skill for your configured AI tools like Claude Code, Cursor, Windsurf, and others. The skill includes component reference, writing standards, and workflow guidance.

See the [AI tools guides](/ai-tools) for tool-specific setup.

## Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mint) to preview your documentation changes locally. To install, use the following command:

```
npm i -g mint
```

Run the following command at the root of your documentation, where your `docs.json` is located:

```
mint dev
```

View your local preview at `http://localhost:3000`.

## API contract maintenance

The checked-in `openapi.json` is derived from Pioneer's complete
`api/openapi.json`. Route-level `x-fastino-visibility: public` metadata in
Pioneer is the only operation-publication decision; docs must not maintain
another allowlist. Refresh and verify it against a local Pioneer checkout:

```bash
uv run --directory ../Pioneer/brain --locked \
  python ../scripts/project_public_openapi.py \
  ../api/openapi.json "$(pwd)/openapi.json"
python3 scripts/check_api_docs.py --pioneer-root ../Pioneer
```

The check also enforces two-way parity with the existing public endpoint pages,
validates the five canonical Mintlify skills, and rejects stale Pioneer API
origins, retired inference routes, skill-install hosts, and doubled `/v1`
prefixes. It also verifies that each published language has a current
`llms.txt` index and that the English root index links to every localized
index. Regenerate localized indexes after changing navigation or page
frontmatter:

```bash
python3 scripts/generate_localized_llms.py
```

## Publishing changes

Install our GitHub app from your [dashboard](https://dashboard.mintlify.com/settings/organization/github-app) to propagate changes from your repo to your deployment. Changes are deployed to production automatically after pushing to the default branch.

## Need help?

### Troubleshooting

- If your dev environment isn't running: Run `mint update` to ensure you have the most recent version of the CLI.
- If a page loads as a 404: Make sure you are running in a folder with a valid `docs.json`.

### Resources
- [Mintlify documentation](https://mintlify.com/docs)
