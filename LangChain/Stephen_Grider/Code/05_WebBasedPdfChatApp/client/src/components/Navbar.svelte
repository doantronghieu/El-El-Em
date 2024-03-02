<script lang="ts">
	import AuthLinks from '$c/auth/AuthLinks.svelte';
	import { auth } from '$s/auth';

	$: user = $auth.user;
</script>

<header
	class="mb-2 drop-shadow flex flex-wrap sm:justify-start sm:flex-nowrap z-50 w-full bg-white text-sm py-4 dark:bg-gray-800"
>
	<nav
		class="max-w-[85rem] w-full mx-auto px-4 sm:flex sm:items-center sm:justify-between"
		aria-label="Global"
	>
		<a class="flex-none text-xl font-semibold dark:text-white" href="/">Home</a>
		<div class="flex flex-row items-center gap-5 mt-5 sm:justify-end sm:mt-0 sm:pl-5">
			<!-- <a
				class="font-medium text-gray-600 hover:text-gray-400 dark:text-gray-400 dark:hover:text-gray-500"
				href="/spec.html">API Spec</a
			> -->
			{#if user}
				<a
					class="font-medium text-gray-600 hover:text-gray-400 dark:text-gray-400 dark:hover:text-gray-500"
					href="/scores">Scores</a
				>
				<a
					class="font-medium text-gray-600 hover:text-gray-400 dark:text-gray-400 dark:hover:text-gray-500"
					href="/documents">Documents</a
				>
			{/if}
			<AuthLinks />
		</div>
	</nav>
</header>
