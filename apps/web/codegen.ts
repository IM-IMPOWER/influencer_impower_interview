import type { CodegenConfig } from '@graphql-codegen/cli'

// AIDEV-NOTE: 250903170004 - GraphQL code generation config for TypeScript type safety

const config: CodegenConfig = {
  schema: process.env.GRAPHQL_SCHEMA_URL || 'http://localhost:8000/graphql',
  documents: ['src/**/*.{ts,tsx}', 'src/lib/graphql/**/*.graphql.ts'],
  generates: {
    './src/__generated__/': {
      preset: 'client',
      plugins: [],
      presetConfig: {
        gqlTagName: 'gql',
        fragmentMasking: false
      }
    },
    './src/__generated__/graphql.ts': {
      plugins: [
        'typescript',
        'typescript-operations',
        'typescript-react-apollo'
      ],
      config: {
        withHooks: true,
        withHOC: false,
        withComponent: false,
        skipTypename: false,
        avoidOptionals: false,
        maybeValue: 'T | null | undefined'
      }
    }
  },
  config: {
    scalars: {
      DateTime: 'string',
      JSON: 'any',
      UUID: 'string'
    }
  }
}

export default config