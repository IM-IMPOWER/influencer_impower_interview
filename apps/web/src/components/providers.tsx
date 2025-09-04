"use client";

import { ApolloProvider } from "@apollo/client";
import { apolloClient } from "@/lib/apollo";
import { ThemeProvider } from "./theme-provider";
import { Toaster } from "./ui/sonner";

// AIDEV-NOTE: 250903170000 - Migrated from tRPC to Apollo GraphQL client for better GraphQL integration

export default function Providers({ children }: { children: React.ReactNode }) {
	return (
		<ThemeProvider
			attribute="class"
			defaultTheme="system"
			enableSystem
			disableTransitionOnChange
		>
			<ApolloProvider client={apolloClient}>
				{children}
			</ApolloProvider>
			<Toaster richColors />
		</ThemeProvider>
	);
}
